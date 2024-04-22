"""
Base class for all ray samplers.
"""

import torch
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked

from ..cameras import Camera
from ..rays import RayBundle, RaySamples


class BaseRaySampler:
    """
    Base class for all ray samplers.
    """

    @jaxtyped(typechecker=typechecked)
    def generate_rays(
        self,
        camera: Camera,
        screen_coords: Int[torch.Tensor, "num_ray 2"],
    ) -> RayBundle:
        """
        Generates rays for the current camera by computing rays' origins and directions.

        Args:
            camera: Current Camera object.
            screen_coords: A flattened array of pixels coordinates in the screen space.

        Returns:
            ray_bundle: An instance of 'RayBundle' containing ray information
                necessary for volume rendering.
        """
        # obtain ray directions in camera frame
        ray_directions = self._get_ray_directions(camera, screen_coords)

        # transform ray directions to world frame
        ray_directions = ray_directions[..., None, :]  # (num_ray, 1, 3)
        camera_orientation = camera.camera_to_world[:3, :3]  # (3, 3)
        ray_directions = torch.sum(ray_directions * camera_orientation, dim=-1)

        # obtain ray origins (camera position) in world frame
        camera_position = camera.camera_to_world[:3, -1]
        ray_origins = camera_position.expand(ray_directions.shape)

        # get near and far bounds
        ones = torch.ones(ray_directions.shape[0], device=camera.device)
        nears = (ones * camera.t_near).unsqueeze(-1)
        fars = (ones * camera.t_far).unsqueeze(-1)

        # pack rays into 'RayBundle'
        return RayBundle(
            origins=ray_origins,  # (num_ray, 3)
            directions=ray_directions,  # (num_ray, 3)
            nears=nears,  # (num_ray)
            fars=fars,  # (num_ray)
        )

    @jaxtyped(typechecker=typechecked)
    def _get_ray_directions(
        self,
        camera: Camera,
        screen_coords: Int[torch.Tensor, "num_ray 2"],
        # TODO: add normalized flag or no?
    ) -> Float[torch.Tensor, "num_ray 3"]:
        """
        Computes view direction vectors represented in the camera frame.

        The direction vectors are represented in the camera frame, which x and y
        axes pointing to the right and up, respectively. The z-axis points
        towards the image (from the camera).

        Args:
            camera: Current Camera object.
            screen_coords: A flattened array of image pixels coordinates.

        Returns:
            ray_directions: Ray directions in the camera frame.
        """
        # list screen (pixel) coordinates: (u, v)
        screen_xs = screen_coords[..., 0].to(camera.device)
        screen_ys = screen_coords[..., 1].to(camera.device)

        # compute ray directions: (u, v) -> (x, y)
        ray_xs = (screen_xs - camera.c_x) / camera.f_x
        ray_ys = -1 * (screen_ys - camera.c_y) / camera.f_y  # opposite y-axis

        # (x, y) -> (x, y, -1)
        ray_zs = -1 * torch.ones_like(ray_xs)  # towards the image
        ray_directions = torch.stack([ray_xs, ray_ys, ray_zs], dim=-1)
        return ray_directions

    def sample_along_rays(self, *args, **kwargs) -> RaySamples:
        """
        Samples points along rays.

        Different type of samplers MUST implement this method.
        """
        raise NotImplementedError()
