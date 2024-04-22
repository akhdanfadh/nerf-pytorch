"""
Scene object representing an renderable scene.
"""

from typing import Sequence, Tuple

import torch

from .primitives.base import BasePrimitive


class Scene:
    """
    Scene object representing an renderable scene.

    Attributes:
        primitives (Sequence[BasePrimitive]): A collection of scene primitives.
    """

    def __init__(self, primitives: Sequence[BasePrimitive]):
        """
        Constructor for 'Scene'.

        Args:
            primitives: A collection of scene primitives.
        """
        self._primitives = primitives

    def query_points(
        self, pos: torch.Tensor, view_dir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query 3D scene to retrieve radiance and density values.

        Args:
            pos: 3D coordinates of sample points.
            view_dir: View direction vectors associated with sample points.

        Returns:
            sigma: The density at each sample point (tensor of shape (N, S)).
            radiance: The radiance at each sample point (tensor of shape (N, S, 3)).
        """
        return self._primitives.query_points(pos, view_dir)
