"""
Ray object handler for representing rays in the scene from a single camera.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
@dataclass
class RayBundle:
    """
    A data structure for representing a collection of rays passing through all
    pixels of an image (one ray per pixel) from a camera.

    Attributes:
        origins (torch.Tensor): Ray origins in the world coordinate.
        directions (torch.Tensor): Ray directions in the world coordinate.
        t_near (torch.Tensor): Near clipping plane.
            The nearest distance rays can reach. (cf. x = o + t_near * d).
        t_far (torch.Tensor): Far clipping plane.
            The farthest distance rays can reach. (cf. x = o + t_far * d).
    """

    origins: Float[torch.Tensor, "*batch_size 3"]
    directions: Float[torch.Tensor, "*batch_size 3"]
    nears: Float[torch.Tensor, "*batch_size 1"]
    fars: Float[torch.Tensor, "*batch_size 1"]

    def __len__(self) -> int:
        """Returns the number of rays in the bundle."""
        return self.origins.shape[0]


@jaxtyped(typechecker=typechecked)
@dataclass(init=False)
class RaySamples:
    """
    A data structure for representing rays with their sample points.

    Attributes:
        ray_bundle (RayBundle): A collection of rays from a camera. Contains ray
            origin, direction, near and far bounds.
        t_samples (torch.Tensor): Distance values sampled along rays.
    """

    ray_bundle: RayBundle
    t_samples: Float[torch.Tensor, "num_ray num_sample"]

    def __init__(
        self,
        ray_bundle: RayBundle,
        t_samples: Float[torch.Tensor, "num_ray num_sample"],
    ):
        """
        Constructor for RaySamples.
        """
        assert len(ray_bundle) == t_samples.shape[0], (
            "Number of rays in ray_bundle and t_samples must match. "
            f"Got {len(ray_bundle)} and {t_samples.shape[0]}, respectively."
        )
        self.ray_bundle = ray_bundle
        self.t_samples = t_samples

    @jaxtyped(typechecker=typechecked)
    def compute_sample_coordinates(self) -> Float[torch.Tensor, "num_ray num_sample 3"]:
        """
        Computes the 3D coordinates of sample points along rays in the ray bundle.

        For a ray r parameterized by the origin o and direction d (not
        necessarily normalized), a point on the ray can be computed as:
        r(t) = o + t * d,
        where t is bounded by the near and far clipping planes.

        Returns:
            sample_coords: 3D coordinates of sample points along rays.
        """
        origin = self.ray_bundle.origins.unsqueeze(1)  # (num_ray, 1, 3)
        direction = self.ray_bundle.directions.unsqueeze(1)  # (num_ray, 1, 3)
        t_sample = self.t_samples.unsqueeze(-1)  # (num_ray, num_sample, 1)

        sample_coords = (
            origin + t_sample * direction
        )  # (num_ray, num_sample, 3) by broadcasting
        return sample_coords

    @jaxtyped(typechecker=typechecked)
    def compute_deltas(
        self, right_end: float = 1e8
    ) -> Float[torch.Tensor, "num_ray num_sample"]:
        """
        Compute differences between adjacent t's required to approximate integrals.

        Args:
            right_end: The value to be appended to the right end when computing
                1st order difference.

        Returns:
            deltas: Differences between adjacent t's. When evaluating the delta
                for the farthest sample on a ray, use the value of the argument
                'right_end'.
        """
        t_samples = self.t_samples
        num_ray = t_samples.shape[0]
        device = t_samples.device

        deltas = torch.diff(
            t_samples,
            n=1,
            dim=-1,
            append=right_end * torch.ones((num_ray, 1), device=device),
        )

        # # alternative implementation
        # deltas = t_samples[..., 1:] - t_samples[..., :-1]
        # deltas = torch.cat([deltas, right_end - t_samples[..., -1:]], dim=-1)

        return deltas
