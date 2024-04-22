"""
Base class for scene primitives.
"""

import warnings
from typing import Dict, Optional, Tuple

import torch

from ...signal_encoder.base import BaseSignalEncoder


class BasePrimitive(object):
    """
    Scene primitive base class.
    """

    def __init__(self, encoders: Optional[Dict[str, BaseSignalEncoder]] = None):
        if encoders is not None:
            if not isinstance(encoders, dict):
                raise ValueError(
                    f"Expected a parameter of type Dict. Got {type(encoders)}"
                )
            if "coord_enc" not in encoders.keys():  # position encoder
                warnings.warn(
                    f"Missing an encoder type 'coord_enc'. Got {encoders.keys()}."
                )
            if "dir_enc" not in encoders.keys():  # direction encoder
                warnings.warn(
                    f"Missing an encoder type 'dir_enc'. Got {encoders.keys()}."
                )
        self._encoders = encoders

    @property
    def encoders(self) -> Optional[Dict[str, BaseSignalEncoder]]:
        """
        Returns the signal encoders that process signals before querying the
        neural radiance field(s).
        """
        return self._encoders

    @encoders.setter
    def encoders(self, new_encoders) -> None:
        """
        Sets (new) signal encoders that process signals before querying the
        neural radiance field(s).
        """
        if not isinstance(new_encoders, dict):
            raise ValueError(
                f"Expected a parameter of type Dict. Got {type(new_encoders)}"
            )
        if "coord_enc" not in new_encoders.keys():
            warnings.warn(
                f"Missing an encoder type 'coord_enc'. Got {new_encoders.keys()}."
            )
        if "dir_enc" not in new_encoders.keys():
            warnings.warn(
                f"Missing an encoder type 'dir_enc'. Got {new_encoders.keys()}."
            )
        self._encoders = new_encoders

    def query_points(
        self, pos: torch.Tensor, view_dir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query 3D scene to retrieve radiance and density values.

        Args:
            pos: 3D coordinates of sample points.
            view_dir: View/ray direction vectors associated with sample points.

        Returns:
            sigma: The density at each sample point (tensor of shape (N, S)).
            radiance: The radiance at each sample point (tensor of shape (N, S, 3)).
        """
        if pos.shape != view_dir.shape:
            raise ValueError(
                "Expected tensors of same shape. "
                f"Got {pos.shape} and {view_dir.shape}, respectively."
            )
        num_ray, num_sample, _ = pos.shape
        return num_ray, num_sample
