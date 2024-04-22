"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple

import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped
from typeguard import typechecked


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(self, pos_dim: int, view_dir_dim: int, feat_dim: int = 256):
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # 8 fully connected layers including a skip connection
        self.mlp_praskip = nn.Sequential(
            nn.Linear(pos_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        )
        self.mlp_postskip = nn.Sequential(
            nn.Linear(feat_dim + pos_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        )

        # predict density
        self.mlp_sigma = nn.Sequential(nn.Linear(feat_dim, 1), nn.ReLU())

        # predict radiance
        self.mlp_bottleneck = nn.Linear(feat_dim, feat_dim)  # orange arrow in the paper
        self.mlp_radiance = nn.Sequential(
            nn.Linear(feat_dim + view_dir_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 3),
            nn.Sigmoid(),
        )

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[
        Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]
    ]:
        """
        Predicts density and radiance (color) of sample points.

        Given sample point coordinates and view directions, this method predicts
        the corresponding density (sigma) and radiance (in RGB) of the sample points.

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The predicted density evaluated at the given sample points.
            radiance: The predicted radiance evaluated at the given sample points.
        """
        feat = self.mlp_praskip(pos)
        feat = self.mlp_postskip(torch.cat([pos, feat], dim=-1))

        sigma = self.mlp_sigma(feat)

        feat = self.mlp_bottleneck(feat)
        radiance = self.mlp_radiance(torch.cat([feat, view_dir], dim=-1))

        return sigma, radiance
