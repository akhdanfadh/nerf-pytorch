"""
Volume renderer introduced in NeRF (ECCV 2020).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked

from ..scene import BasePrimitive
from .cameras import Camera
from .integrator import BaseIntegrator
from .ray_sampler import BaseRaySampler
from .rays import RayBundle, RaySamples


@dataclass(init=False)
class VolumeRenderer:
    """
    Volume renderer introduced in NeRF (ECCV 2020).

    Attributes:
        integrator: An instance of class derived from 'BaseIntegrator'.
            Computes numerical integrations to determine pixel colors in a differentiable manner.
        sampler: An instance of class derived from 'BaseRaySampler'.
            Samples the points in 3D space to evaluate neural scene representations.
        pixel_indices: Indices of pixels from where rays will be casted.
    """

    sampler: BaseRaySampler
    integrator: BaseIntegrator

    def __init__(self, sampler: BaseRaySampler, integrator: BaseIntegrator):
        """
        Constructor of class 'VolumeRenderer'.
        """
        self.sampler = sampler
        self.integrator = integrator
        self.pixel_indices: Optional[Int[torch.Tensor, "num_ray"]] = None

    @jaxtyped(typechecker=typechecked)
    def render_scene(
        self,
        target_scene: BasePrimitive,
        camera: Camera,
        num_samples: int,
        num_ray_batch: int = 1,
        num_pixels: Optional[int] = None,
        pixel_indices: Optional[Int[torch.Tensor, "num_ray"]] = None,
        prev_weights: Optional[Float[torch.Tensor, "num_ray prev_num_sample"]] = None,
        prev_t_samples: Optional[Float[torch.Tensor, "num_ray prev_num_sample"]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "num_ray 3"],
        Float[torch.Tensor, "num_ray num_sample"],
        Float[torch.Tensor, "num_ray num_sample"],
    ]:
        """
        Renders the scene by querying underlying 3D inductive bias.

        Args:
            target_scene: An instance of class derived from 'BasePrimitive'.
                Represents the 3D scene to be rendered.
            camera: An instance of class 'Camera'.
                Represents the camera parameters.
            num_pixels: Number of pixels to render.
            num_samples: Number of samples drawn along each ray.
            pixel_indices: Indices of pixels from where rays will be casted.
            prev_weights: If provided, hierarchical sampling is being performed.
                The weights are obtained by evaluating the 'coarse' network.
            prev_t_samples: Previous sample points to be used for hierarchical sampling.
            num_ray_batch: Number of batches to divide the entire set of rays.
                It is necessary to avoid out-of-memory during rendering.

        Returns:
            pixel_rgb: The final pixel colors of rendered image lying in RGB color space.
            weights: Weight of each sample point along rays.
                For hierarchical sampling, num_sample = prev_num_sample + num_samples.
            t_samples: Sample points along rays.
                For hierarchical sampling, num_sample = prev_num_sample + num_samples.
        """

        # sample pixels from which rays will be casted
        if pixel_indices is None:
            pixel_indices = self._get_pixel_indices(
                camera.image_height.item(), camera.image_width.item(), num_pixels
            )
        self.pixel_indices = pixel_indices

        # obtain rays to render
        coords_to_render: Int[torch.Tensor, "num_ray 2"] = camera.screen_coords[
            pixel_indices, :
        ]
        ray_bundle: RayBundle = self.sampler.generate_rays(camera, coords_to_render)

        # =====================================================================
        # Memory-bandwidth intensive operations (must be done directly on GPUs)
        # =====================================================================

        # sample points along rays
        ray_samples: RaySamples = self.sampler.sample_along_rays(
            ray_bundle,
            num_samples,
            prev_weights,
            prev_t_samples,
        )

        # render rays
        pixel_rgb, prev_weights, _, _ = self._render_ray_batches(
            target_scene,
            ray_samples,
            num_ray_batch,
        )

        # =====================================================================
        # Memory-bandwidth intensive operations (must be done directly on GPUs)
        # =====================================================================

        return pixel_rgb, prev_weights, ray_samples.t_samples

    @jaxtyped(typechecker=typechecked)
    def _get_pixel_indices(
        self, img_height: int, img_width: int, num_pixels: Optional[int] = None
    ) -> Int[torch.Tensor, "num_ray"]:
        """
        Randomly samples pixels from the image to render.

        Args:
            img_height: Height of the image.
            img_width: Width of the image.
            num_pixels: Number of pixels to render.

        Returns:
            pixel_indices: Indices of pixels to render.
        """
        total_pixels = img_height * img_width
        if num_pixels is None:
            output_size = total_pixels
        else:
            output_size = num_pixels if num_pixels < total_pixels else total_pixels

        pixel_indices = torch.tensor(
            np.random.choice(
                total_pixels,  # choose from np.arange(total_pixels)
                size=output_size,  # output shape
                replace=False,  # no duplicates
            )
        )

        return pixel_indices

    @jaxtyped(typechecker=typechecked)
    def _render_ray_batches(
        self,
        target_scene: BasePrimitive,
        ray_samples: RaySamples,
        num_batch: int = 1,
    ) -> Tuple[
        Float[torch.Tensor, "num_ray 3"],
        Float[torch.Tensor, "num_ray num_sample"],
        Float[torch.Tensor, "num_ray num_sample"],
        Float[torch.Tensor, "num_ray num_sample 3"],
    ]:
        """
        Renders an image by dividing its pixels into small batches.

        Args:
            target_scene: An instance of class derived from 'BasePrimitive'.
                Represents the 3D scene to be rendered.
            ray_samples: An instance of class 'RaySamples'.
                Contains all rays with their sample points.
            num_batch: Number of batches to divide the entire set of rays.

        Returns:
            pixel_rgb: The pixel intensities determined by integrating radiance
                values along each ray.
            weights: The weight obtained by multiplying transmittance and alpha
                during numerical integration.
            sigma: Estimated densities at sample points.
            radiance: Estimated radiance values at sample points.
        """
        rgb, weights, sigma, radiance = [], [], [], []

        sample_pts: Float[torch.Tensor, "num_ray num_sample 3"] = (
            ray_samples.compute_sample_coordinates()
        )
        ray_dir: Float[torch.Tensor, "num_ray 3"] = ray_samples.ray_bundle.directions
        delta_t: Float[torch.Tensor, "num_ray 3"] = ray_samples.compute_deltas()

        # multiply ray direction norms to compute distance between consecutive sample points
        dir_norm = torch.norm(ray_dir, dim=1, keepdim=True)  # length of vector
        delta_t = delta_t * dir_norm

        pts_chunks = torch.chunk(sample_pts, num_batch, dim=0)
        dir_chunks = torch.chunk(ray_dir, num_batch, dim=0)
        delta_chunks = torch.chunk(delta_t, num_batch, dim=0)
        assert len(pts_chunks) == len(dir_chunks) == len(delta_chunks), (
            "Lenghts of pts_chunks, dir_chunks, and delta_chunks must match. "
            f"Got {len(pts_chunks)}, {len(dir_chunks)}, and {len(delta_chunks)}, respectively."
        )

        for pts_batch, dir_batch, delta_batch in zip(
            pts_chunks, dir_chunks, delta_chunks
        ):
            # query the scene to get density and radiance
            sigma_batch, radiance_batch = target_scene.query_points(
                pts_batch, dir_batch
            )

            # compute pixel colors by evaluating the volume rendering equation
            rgb_batch, weights_batch = self.integrator.integrate_along_rays(
                sigma_batch, radiance_batch, delta_batch
            )

            # collect rendering outputs
            rgb.append(rgb_batch)
            weights.append(weights_batch)
            sigma.append(sigma_batch)
            radiance.append(radiance_batch)

        pixel_rgb = torch.cat(rgb, dim=0)
        weights = torch.cat(weights, dim=0)
        sigma = torch.cat(sigma, dim=0)
        radiance = torch.cat(radiance, dim=0)

        return pixel_rgb, weights, sigma, radiance
