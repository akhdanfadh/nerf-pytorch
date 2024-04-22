"""
Implementation of stratified sampling in NeRF.
"""

from typing import Optional, Union

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from ..rays import RayBundle, RaySamples
from .base import BaseRaySampler


class StratifiedSampler(BaseRaySampler):
    """
    Stratified sampler proposed in NeRF (ECCV 2020).
    """

    @jaxtyped(typechecker=typechecked)
    def sample_along_rays(
        self,
        ray_bundle: RayBundle,
        num_samples: int,
        prev_weights: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
        prev_t_samples: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
    ) -> RaySamples:
        """
        Samples points along rays.

        Args:
            ray_bundle: An instance of RayBundle containing ray origin, direction,
                and the nearest/farthest sample distances.
            num_samples: Number of samples drawn along each ray.
            prev_weights: If provided, hierarchical sampling is being performed.
            prev_t_samples: Previous sample points to be used for hierarchical sampling.

        Returns:
            ray_samples: An instance of RaySamples containing the sampled points along rays.
        """
        if prev_weights is None:
            t_samples = self._sample_along_rays_uniform(
                num_ray=len(ray_bundle),
                num_sample=num_samples,
                near=ray_bundle.nears[0].item(),
                far=ray_bundle.fars[0].item(),
                device=ray_bundle.origins.device,
            )
        else:
            assert (
                prev_t_samples is not None
            ), "Previous scene's t_samples must be provided."
            t_samples = self._sample_along_rays_importance(
                prev_weights, prev_t_samples, num_samples
            )

        ray_samples = RaySamples(ray_bundle, t_samples)
        return ray_samples

    @jaxtyped(typechecker=typechecked)
    def _sample_along_rays_uniform(
        self,
        num_ray: int,
        num_sample: int,
        near: float,
        far: float,
        device: Union[str, torch.device],
    ) -> Float[torch.Tensor, "num_ray num_sample"]:
        """
        Performs uniform sampling of points along rays.

        Given the nearest and farthest scene bound t_near and t_far, the algorithm:
        (1) partitions the interval [t_near, t_far] into num_sample bins of equal length;
        (2) draws one sample from the uniform distribution within each bin;
        Refer to section 4 in the paper for details.

        Args:
            num_ray: The number of rays to sample points along.
            num_sample: The number of samples to be generated along each ray.
            near: The nearest distance rays can reach.
            far: The farthest distance rays can reach.
            device: The index of CUDA device where results of ray sampling will be located.

        Returns:
            t_samples: The distance values sampled along rays. The values should
                lie in the range defined by the near and far bounds.
        """
        # equally partition the interval [0.0, 1.0] inclusively, for efficient compute
        t_bins = torch.linspace(0.0, 1.0, num_sample + 1, device=device)
        bin_size = t_bins[1] - t_bins[0]

        # arrange the start of each bin for each ray
        t_bins_start = t_bins[:-1].unsqueeze(0)  # (1, num_sample)
        t_bins_start = t_bins_start.repeat(num_ray, 1)  # (num_ray, num_sample)

        # sample from the uniform distribution within each bin and scale to [t_near, t_far]
        t_samples = t_bins_start + bin_size * torch.rand_like(t_bins_start)
        t_samples = near * (1.0 - t_samples) + far * t_samples
        return t_samples

    @jaxtyped(typechecker=typechecked)
    def _sample_along_rays_importance(
        self,
        prev_weights: Float[torch.Tensor, "num_ray num_sample"],
        prev_t_samples: Float[torch.Tensor, "num_ray num_sample"],
        num_sample: int,
    ) -> Float[torch.Tensor, "num_ray new_num_sample"]:
        """
        Draws samples from the probability density represented by given weights.

        Performs the inverse CDF sampling of points along rays given weights
        indicating the 'importance' of each given sample. It works as follows:
        (1) Normalize the weights to form a piecewise constant PDF.
        (2) Sample points from the PDF using the inverse CDF sampling.
        Refer to section 5.2 in the paper for details.

        Args:
            prev_weights: Weights obtained by evaluating the 'coarse' network.
                An unnormalized PDF is represented as a vector of shape (num_sample,).
            prev_t_samples: Previous sample points on the 'coarse' network.
                Note that the elements are assumed to be ordered.
            num_sample: The number of samples to be generated along each ray.

        Returns:
            t_samples: The distance values sampled along rays.
        """
        # compute midpoints of intervals and weights for the midpoints
        t_mid = 0.5 * (prev_t_samples[..., 1:] + prev_t_samples[..., :-1])
        weights_mid = prev_weights[..., 1:-1]

        # construct the PDF
        weights_mid += 1e-5
        normalizer = torch.sum(weights_mid, dim=-1, keepdim=True)
        pdf = weights_mid / normalizer

        # compute the CDF
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

        # sample from the uniform distribution: U[0, 1)
        cdf_ys = torch.rand(list(cdf.shape[:-1]) + [num_sample], device=cdf.device)
        cdf_ys = cdf_ys.contiguous()

        # inverse CDF sampling
        indices = torch.searchsorted(cdf, cdf_ys, right=True)
        lower = torch.max(torch.zeros_like(indices - 1), indices - 1)
        upper = torch.min(cdf.shape[-1] - 1 * torch.ones_like(indices), indices)

        cdf_lower = torch.gather(cdf, 1, lower)
        cdf_upper = torch.gather(cdf, 1, upper)
        bins_lower = torch.gather(t_mid, 1, lower)
        bins_upper = torch.gather(t_mid, 1, upper)

        # approximate the CDF to a linear function within each interval
        denom = cdf_upper - cdf_lower
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (
            cdf_ys - cdf_lower
        ) / denom  # i.e., cdf_y = cdf_lower + t * (cdf_upper - cdf_lower)
        new_t_samples = bins_lower + t * (bins_upper - bins_lower)
        assert torch.isnan(new_t_samples).sum() == 0

        # combine the new samples with the previous ones
        t_samples = torch.cat([prev_t_samples, new_t_samples], -1)
        t_samples, _ = torch.sort(t_samples, dim=-1)

        return t_samples
