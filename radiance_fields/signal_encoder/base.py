"""
Base class for signal encoders.
"""

import torch


class BaseSignalEncoder:
    """
    Base class for signal encoders.
    """

    def __init__(self):
        pass

    def encode(self, in_signal: torch.Tensor) -> torch.Tensor:
        """
        Computes the encoding of the given signal.

        Args:
            in_signal: Input signal (tensor of shape(N, C)) being encoded.

        Returns:
            Encoded signal (tensor of shape(N, self.out_dim)).
        """
        raise NotImplementedError()
