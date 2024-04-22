"""
Implementation of positional encoding.
"""

import torch

from .base import BaseSignalEncoder


class PositionalEncoder(BaseSignalEncoder):
    """
    Implementation of positional encoding.

    Attributes:
        in_dim (int): Dimensionality of the data.
        embed_level (int): Level of positional encoding.
        out_dim (int): Dimensionality of the encoded data.
    """

    def __init__(self, in_dim: int, embed_level: int, include_input: bool):
        """
        Constructor for PositionalEncoder.

        Args:
            in_dim (int): Dimensionality of the data.
            embed_level (int): Level of positional encoding.
            include_input (bool): Whether to include raw input in the encoding.
        """
        super().__init__()

        self._in_dim = in_dim
        self._embed_level = embed_level
        self._include_input = include_input

        # encode to higher dimensional space R^{2L}
        self._out_dim = self._in_dim * 2 * self._embed_level
        if self._include_input:
            self._out_dim += self._in_dim

        # creating embedding function
        self._embed_fns = self._create_embedding_fn()

    @property
    def in_dim(self) -> int:
        """
        Returns the dimensionality of the input vector that the encoder takes.
        """
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """
        Returns the dimensionality of the output vector after encoding.
        """
        return self._out_dim

    def _create_embedding_fn(self):
        """
        Creates embedding function from given
            (1) number of frequency bands;
            (2) dimension of data being encoded;

        The positional encoding is defined as (see section 5.1 in the paper):
        f(p) = [
                sin(2^0 * pi * p), cos(2^0 * pi * p),
                                ...,
                sin(2^{L-1} * pi * p), cos(2^{L-1} * pi * p)
            ],
        and is computed for all components of the input vector.
        """
        max_freq_level = self._embed_level
        freq_bands = torch.pow(2.0, torch.arange(max_freq_level, dtype=torch.float32))

        embed_fns = []
        if self._include_input:
            embed_fns.append(lambda x: x)
        for freq in freq_bands:
            embed_fns.append(lambda x, freq=freq: torch.sin(freq * x))
            embed_fns.append(lambda x, freq=freq: torch.cos(freq * x))

        return embed_fns

    def encode(self, in_signal: torch.Tensor) -> torch.Tensor:
        """
        Computes positional encoding of the given signal.

        Args:
            in_signal: Input signal (tensor of shape(N, C)) being encoded.

        Returns:
            Positional encoded signal (tensor of shape(N, self.out_dim)).
        """
        return torch.cat([fn(in_signal) for fn in self._embed_fns], dim=-1)
