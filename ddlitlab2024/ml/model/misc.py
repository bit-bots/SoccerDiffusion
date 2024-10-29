import numpy as np
import torch
from torch import nn


class StepToken(nn.Module):
    """
    Encodes the current step of the diffusion process as a token that can be added to the context.\
    It is build from a sinusoidal positional encoding which is concatenated with a learnable token.
    The learnable parameters are introduced to not interfere with the sinusoidal positional encoding used for other tokens in the context.
    This way they can be separated by the attention mechanism.
    """
    def __init__(self, dim):
        """
        Initializes the StepToken module.

        :param dim: The number of hidden dimensions used for the tokens of the parent model.
        """
        super().__init__()
        self.dim = dim
        self.token = nn.Parameter(torch.randn(1, dim // 2))

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        """
        Embeds the current step of the diffusion process as a token.

        :param steps: The current step of the diffusion process (each element of the batch has its own step).
        """
        half_dim = self.dim // 4
        emb = torch.exp(torch.arange(half_dim, device=steps.device) * -np.log(10000) / (half_dim - 1))
        emb = steps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos(), self.token.expand((steps.size(0), self.dim // 2))), dim=-1).unsqueeze(1)
        return emb


class PositionalEncoding(nn.Module):
    """
    A standard positional encoding module for the Transformer model.
    """
    def __init__(self, d_model, max_len):
        """
        Initializes the PositionalEncoding module.

        :param d_model: The number of hidden dimensions used for the tokens of the parent model.
        :param max_len: The maximum length of the input sequences.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        """
        Adds the positional encoding to the input tensor.

        :param x: The input tensor.
        :return: The input tensor with the positional encoding added.
        """
        return x + self.pe[:, : x.size(1)].to(x.device)