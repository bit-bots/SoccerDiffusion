import torch
from torch import nn

from ddlitlab2024.ml.model.decoder import DiffusionActionGenerator
from ddlitlab2024.ml.model.encoder.action_history import ActionHistoryEncoder
from ddlitlab2024.ml.model.misc import StepToken


class End2EndDiffusionTransformer(nn.Module):
    def __init__(
        self, num_joints, hidden_dim, num_layers, num_heads, max_action_context_length, trajectory_prediction_length
    ):
        super().__init__()
        self.step_encoding = StepToken(hidden_dim)
        self.action_history_encoder = ActionHistoryEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_action_context_length,
        )
        self.diffusion_action_generator = DiffusionActionGenerator(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=trajectory_prediction_length,
        )

        # Store normalization parameters
        self.register_buffer("mean", torch.zeros(num_joints))
        self.register_buffer("std", torch.ones(num_joints))

    def forward(self, past_actions, noisy_action_predictions, step):
        # Encode the past actions
        context = self.action_history_encoder(past_actions)  # This can be cached during inference TODO
        # Add additional information to the context
        # Generate step token to encode the current step of the diffusion process
        step_token = self.step_encoding(step)
        context = torch.cat([step_token, context], dim=1)
        # Denoise the noisy action predictions
        return self.diffusion_action_generator(noisy_action_predictions, context)
