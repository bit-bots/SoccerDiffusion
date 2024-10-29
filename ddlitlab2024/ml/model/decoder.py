from torch import nn
from ddlitlab2024.ml.model.misc import PositionalEncoding


class DiffusionActionGenerator(nn.Module):
    """
    The DiffusionActionGenerator module is a transformer decoder that takes the noisy action predictions and
    the context (past actions, sensor data, etc.) as input and outputs the denoised action predictions.
    """
    def __init__(self, num_joints, hidden_dim, num_layers, num_heads, max_seq_len):
        """
        Initializes the DiffusionActionGenerator module.

        :param num_joints: The number of joints in the robot.
        :param hidden_dim: The number of hidden dimensions.
        :param num_layers: The number of transformer layers.
        :param num_heads: The number of attention heads.
        :param max_seq_len: The maximum length of the input sequences (used for positional encoding).
        """
        super().__init__()
        self.embedding = nn.Linear(num_joints, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(hidden_dim, num_joints)

    def forward(self, x, context):
        """
        Forward pass of the DiffusionActionGenerator module.

        :param x: The noisy action predictions. Shape: (batch_size, seq_len, joint)
        :param context: The context (diffusion step, past actions, sensor data, etc.). Shape: (batch_size, seq_len, hidden_dim)
        :return: The denoised action predictions. Shape: (batch_size, seq_len, joint)
        """
        # Embed the input
        x = self.embedding(x)
        # Positional encoding
        x = self.positional_encoding(x)
        # Pass through the transformer decoder
        out = self.transformer_decoder(x, context)
        # Final projection layer
        return self.fc_out(out)