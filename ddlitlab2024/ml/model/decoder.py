from torch import nn

from ddlitlab2024.ml.model.misc import PositionalEncoding


class DiffusionActionGenerator(nn.Module):
    """
    The DiffusionActionGenerator module is a MLP decoder that takes the noisy action predictions and
    the context (past actions, sensor data, etc.) as input and outputs the denoised action predictions.
    """

    def __init__(self, num_joints, hidden_dim, seq_len):
        """
        Initializes the DiffusionActionGenerator module.

        :param num_joints: The number of joints in the robot.
        :param hidden_dim: The number of hidden dimensions.
        :param seq_len: The length of the sequence.
        """
        super().__init__()
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.embedding = nn.Linear(num_joints * seq_len, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_joints * seq_len)

    def forward(self, x, context):
        """
        Forward pass of the DiffusionActionGenerator module.

        :param x: The noisy action predictions. Shape: (batch_size, seq_len, joint)
        :param context: The context (diffusion step, past actions, sensor data, etc.).
            Shape: (batch_size, hidden_dim)
        :return: The denoised action predictions. Shape: (batch_size, seq_len, joint)
        """
        # Embed the input
        x = self.activation(self.embedding(x.view(x.size(0), self.num_joints * self.seq_len)))
        # Add the context
        x += context
        # Apply hidden layer
        x = self.activation(self.hidden_layer(x))
        # Final projection layer
        return self.fc_out(x).view(x.size(0), self.seq_len, self.num_joints)
