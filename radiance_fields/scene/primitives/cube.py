"""
A simple cubic scene primitive for forward-facing, bounded scenes.
"""

from typing import Dict, Optional, Tuple

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from ...signal_encoder.base import BaseSignalEncoder
from .base import BasePrimitive


class CubePrimitive(BasePrimitive):
    """
    A simple cubic scene primitive.

    Attributes:
        radiance_field (torch.nn.Module): A network representing the scene.
    """

    def __init__(
        self,
        radiance_field: torch.nn.Module,
        encoders: Optional[Dict[str, BaseSignalEncoder]] = None,
    ):
        """
        Constructor for 'PrimitiveCube'.

        Args:
            radiance_field: A network representing the scene.
        """
        super().__init__(encoders=encoders)

        if not isinstance(radiance_field, torch.nn.Module):
            raise ValueError(
                f"Expected a parameter of type torch.nn.Module. Got {type(radiance_field)}."
            )
        self._radiance_field = radiance_field

    @property
    def radiance_field(self) -> torch.nn.Module:
        """
        Returns the network queried through this query structure.
        """
        return self._radiance_field

    @jaxtyped(typechecker=typechecked)
    def query_points(
        self,
        pos: Float[torch.Tensor, "num_ray num_sample 3"],
        view_dir: Float[torch.Tensor, "num_ray 3"],
    ) -> Tuple[
        Float[torch.Tensor, "num_ray num_sample"],
        Float[torch.Tensor, "num_ray num_sample 3"],
    ]:
        """
        Queries the volume bounded by the cube to retrieve radiance and density values.

        Args:
            pos: 3D coordinates of sample points.
            view_dir: View direction vectors associated with sample points.

        Returns:
            sigma: The density at each sample point.
            radiance: The radiance at each sample point.
        """
        # retrieve the number of rays and samples
        num_ray, num_sample, _ = pos.shape

        # handle view direction
        view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True)  # normalize
        view_dir = view_dir.unsqueeze(1)  # add a dimension -> (., 1, .)
        view_dir = view_dir.repeat(1, num_sample, 1)  # repeat -> (., num_sample, .)

        if self.encoders is not None:  # encode input signals
            if "coord_enc" in self.encoders.keys():
                pos = self.encoders["coord_enc"].encode(
                    pos.reshape(num_ray * num_sample, -1)
                )
            if "dir_enc" in self.encoders.keys():
                view_dir = self.encoders["dir_enc"].encode(
                    view_dir.reshape(num_ray * num_sample, -1)
                )

        # query the radiance field
        sigma, radiance = self.radiance_field(pos, view_dir)

        # reshape back the output tensors
        sigma = sigma.reshape(num_ray, num_sample)
        radiance = radiance.reshape(num_ray, num_sample, -1)

        return sigma, radiance
