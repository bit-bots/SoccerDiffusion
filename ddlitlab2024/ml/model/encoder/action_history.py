import torch
from torch import nn
from ddlitlab2024.ml.model.misc import PositionalEncoding

class ActionHistoryEncoder(nn.Module):
    """
    Transformer encoder that encodes the action history of the robot.
    """
    def __init__(self, num_joints, hidden_dim, num_layers, num_heads, max_seq_len):
        """
        Initializes the module.

        :param num_joints: The number of joints in the robot.
        :param hidden_dim: The number of hidden dimensions.
        :param num_layers: The number of transformer layers.
        :param num_heads: The number of attention heads.
        :param max_seq_len: The maximum length of the input sequences (used for positional encoding
        """
        super().__init__()
        self.embedding = nn.Linear(num_joints, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            num_layers=num_layers,
        )

    def forward(self, past_actions: torch.Tensor) -> torch.Tensor:
        """
        Encodes the past actions of the robot as context tokens.

        :param past_actions: The past actions of the robot. Shape: (batch_size, seq_len, joint)
        :return: The encoded context tokens. Shape: (batch_size, seq_len, hidden_dim)
        """
        x = past_actions
        # Embed the input
        x = self.embedding(x)
        # Positional encoding
        x = self.positional_encoding(x)
        # Pass through the transformer encoder
        return self.transformer_encoder(x)