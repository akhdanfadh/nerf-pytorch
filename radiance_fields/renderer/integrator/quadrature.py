"""
Integrator implementing quadrature rule.
"""

from typing import Tuple

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from .base import BaseIntegrator


class QuadratureIntegrator(BaseIntegrator):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped(typechecker=typechecked)
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[
        Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]
    ]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for direct
        volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Final pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        sigma_delta = sigma * delta

        # compute transmittance: T_i = exp(-sigma_j * delta_j)
        zeros = torch.zeros(
            (sigma.shape[0], 1), device=sigma_delta.device
        )  # (num_ray, 1)
        zeros_cat = torch.cat([zeros, sigma_delta], dim=-1)  # (num_ray, num_sample+1)
        transmittance = torch.exp(
            -1 * torch.cumsum(zeros_cat, dim=-1)[..., :-1]
        )  # (num_ray, num_sample)

        # compute alpha: (1 - exp(-sigma_i * delta_i))
        alpha = 1.0 - torch.exp(-sigma_delta)

        # compute weight: w_i = T_i * alpha
        weights = transmittance * alpha

        # compute numerical integral to determine pixel colors:
        # C = sum_{i=1}^{num_sample} T_i * alpha_i * c_i
        weighted_radiance = weights.unsqueeze(-1) * radiance  # (num_ray, num_sample, 3)
        rgbs = torch.sum(weighted_radiance, dim=-2)  # (num_ray, 3)

        return rgbs, weights
