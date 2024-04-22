"""
Camera class used inside renderer(s).

Useful articles on understanding camera parameters and many coordinate frames:
- https://www.baeldung.com/cs/focal-length-intrinsic-camera-parameters
- https://ksimek.github.io/2013/08/13/intrinsic/
- https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf
- https://www.cse.psu.edu/~rtc12/CSE486/lecture13.pdf
- http://www.codinglabs.net/article_world_view_projection_matrix.aspx
- https://stackoverflow.com/questions/695043/how-does-one-convert-world-coordinates-to-camera-coordinates
"""

from dataclasses import dataclass
from typing import Union

import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from typeguard import typechecked


@dataclass(init=False)
class Camera:
    """
    Camera class used inside renderer(s).

    Attributes:
        camera_to_world (torch.Tensor): Camera-to-World matrix of shape [3, 4].
            Calculate where in the world a specific camera space point is.
        f_x (torch.Tensor): Focal length along the x-axis.
        f_y (torch.Tensor): Focal length along the y-axis.
        c_x (torch.Tensor): Principal point along the x-axis.
        c_y (torch.Tensor): Principal point along the y-axis.
        t_near (torch.Tensor): Near clipping plane.
            The nearest distance rays can reach. (cf. x = o + t_near * d).
        t_far (torch.Tensor): Far clipping plane.
            The farthest distance rays can reach. (cf. x = o + t_far * d).
        image_width (torch.Tensor): Image width.
        image_height (torch.Tensor): Image height.
        screen_coords (torch.Tensor): Screen coordinates corresponding to image pixels.
        device (Union[str, torch.device]): Device where camera information is stored.
    """

    camera_to_world: Float[Tensor, "3 4"]
    f_x: Float[Tensor, "1"]
    f_y: Float[Tensor, "1"]
    c_x: Float[Tensor, "1"]
    c_y: Float[Tensor, "1"]
    t_near: Float[Tensor, "1"]
    t_far: Float[Tensor, "1"]
    image_width: Int[Tensor, "1"]
    image_height: Int[Tensor, "1"]
    screen_coords: Int[torch.Tensor, "total_pixel 2"]
    device: Union[str, torch.device]

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        camera_to_world: Float[Tensor, "3 4"],
        f_x: Union[float, Float[Tensor, "1"]],
        f_y: Union[float, Float[Tensor, "1"]],
        c_x: Union[float, Float[Tensor, "1"]],
        c_y: Union[float, Float[Tensor, "1"]],
        near: Union[float, Float[Tensor, "1"]],
        far: Union[float, Float[Tensor, "1"]],
        image_width: Union[int, Int[Tensor, "1"]],
        image_height: Union[int, Int[Tensor, "1"]],
        device: Union[str, torch.device],
    ):
        """
        Constructor for Camera.
        """
        self.camera_to_world = camera_to_world.clone().detach().to(device)
        self.f_x = torch.tensor(f_x, device=device, dtype=torch.float)
        self.f_y = torch.tensor(f_y, device=device, dtype=torch.float)
        self.c_x = torch.tensor(c_x, device=device, dtype=torch.float)
        self.c_y = torch.tensor(c_y, device=device, dtype=torch.float)
        self.t_near = torch.tensor(near, device=device, dtype=torch.float)
        self.t_far = torch.tensor(far, device=device, dtype=torch.float)
        self.image_width = torch.tensor(image_width, device=device, dtype=torch.int)
        self.image_height = torch.tensor(image_height, device=device, dtype=torch.int)
        self.device = device

        # precompute screen coordinates
        self.screen_coords = self._compute_screen_coordinates()

    @jaxtyped(typechecker=typechecked)
    def _compute_screen_coordinates(self) -> Int[torch.Tensor, "total_pixel 2"]:
        """
        Generates screen space coordinates corresponding to whole image pixels.

        The origin of the coordinate frame is located at the top left corner of
        an image, with the x and y axes pointing to the right and down, respectively.

        Returns:
            screen_coords: Screen coordinates corresponding to image pixels.
        """
        img_height = self.image_height.item()
        img_width = self.image_width.item()

        # generate pixel grid
        y_indices = torch.arange(img_height)
        x_indices = torch.arange(img_width)
        y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing="ij")

        # convert to screen coordinates
        screen_coords = torch.stack([x_grid, y_grid], dim=-1)
        screen_coords = screen_coords.reshape(img_height * img_width, 2)

        # # alternative implementation
        # i_grid, j_grid = i_grid.flatten(), j_grid.flatten()
        # screen_coords = torch.stack([j_grid, i_grid], dim=-1)

        return screen_coords
