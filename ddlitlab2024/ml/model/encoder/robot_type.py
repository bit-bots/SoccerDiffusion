import torch
from torch import nn


class RobotTypeEncoder(nn.Module):
    """
    Embeds the robot type into learned context tokens.
    """

    def __init__(self, hidden_dim: int):
        """
        Initializes the module.
        """
        super().__init__()
        NUM_ROBOT_TYPES = 2  # Number of robot types
        self.embedding = nn.Embedding(NUM_ROBOT_TYPES, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input vectors into context tokens.

        :param x: The input states. Shape: (batch_size, NUM_ROBOT_TYPES)
        :return: The encoded context tokens. Shape: (batch_size, hidden_dim)
        """
        # Embed the input
        return self.embedding(x).unsqueeze(1)
