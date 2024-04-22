"""
Base class for integrators.
"""

import torch


class BaseIntegrator:
    """
    Base class for integrators.
    """

    def integrate_along_rays(
        self,
        sigma: torch.Tensor,
        radiance: torch.Tensor,
        delta: torch.Tensor,
    ):
        """
        Determines pixel colors given densities, interval length, and radiance
        values obtained along rays.
        """
        raise NotImplementedError()
