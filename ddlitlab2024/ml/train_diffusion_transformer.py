import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from tqdm import tqdm

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrajectoryTransformerModel(nn.Module):
    def __init__(self, num_joints, hidden_dim, num_layers, num_heads, max_seq_len):
        super().__init__()
        self.embedding = nn.Linear(num_joints, hidden_dim)
        self.merge_embedding = nn.Linear(hidden_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.step_encoding = StepEncoding(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(hidden_dim, num_joints)

    def forward(self, x, step):
        # x shape: (batch_size, seq_len, joint, num_bins)
        # Flatten the joint and bin dimensions into a single token dimension
        x = x.view(x.size(0), x.size(1), -1)
        # Positional encoding
        x = self.embedding(x) + self.positional_encoding(x)
        # Merge embedding
        x = self.merge_embedding(x)
        # Add a step encoding
        x = self.step_encoding(x, step)
        # Create a causal mask
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        # Memory tensor (not used)
        memory = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)
        # Pass through the transformer decoder
        out = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)  # Causal mask applied
        # Final classification layer (logits for each bin)
        return self.fc_out(out)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


# Positional Encoding class for the Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, : x.size(1)].to(x.device)


# Sinosoidal step encoding
class StepEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # div terms should not be trainable
        self.div_term = torch.exp(
            torch.arange(0, d_model, 2, requires_grad=False).float() * (-np.log(10000.0) / d_model)
        ).to(device)
        self.d_model = d_model

    def forward(self, x, step):
        x[..., 0::2] += torch.sin(step.unsqueeze(1) * self.div_term).unsqueeze(1)
        x[..., 1::2] += torch.cos(step.unsqueeze(1) * self.div_term).unsqueeze(1)
        return x


# Define dimensions for the Transformer model
trajectory_dim = 1  # 1D input for the sine wave
hidden_dim = 512
num_layers = 2
num_heads = 4
sequence_length = 30

# Generate a dataset of sine wave trajectories (500 samples)
num_samples = 10000
time = torch.linspace(0, 2 * np.pi, sequence_length).unsqueeze(-1).to(device)
real_trajectories = torch.sin(time + torch.rand(1, num_samples).to(device) * 2 * np.pi).permute(1, 0).to(device)

# Plot the first 5 sine wave trajectories
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(time.cpu(), real_trajectories[i].cpu(), label=f"Trajectory {i + 1}")
plt.title("Sine Wave Trajectories")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

epochs = 100
batch_size = 32

# Initialize the Transformer model and optimizer, and move model to device
model = TrajectoryTransformerModel(
    num_joints=trajectory_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    max_seq_len=sequence_length,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs * (num_samples // 32))

scheduler = DDPMScheduler()
scheduler.config.num_train_timesteps = 1000


# Training loop
for epoch in tqdm(range(epochs)):  # Number of training epochs
    for batch in range(num_samples // batch_size):
        targets = real_trajectories[batch * batch_size : (batch + 1) * batch_size].to(device)

        optimizer.zero_grad()

        # Sample a random timestep for each trajectory in the batch
        random_timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,)).long().to(device)

        # Sample noise to add to the entire trajectory
        noise = torch.randn_like(targets).to(device)

        # Forward diffusion: Add noise to the entire trajectory at the random timestep
        noisy_trajectory = scheduler.add_noise(targets, noise, random_timesteps)

        # Plot the noisy trajectory and the original trajectory
        # plt.figure(figsize=(12, 6))
        # plt.plot(time.cpu(), targets[0].cpu(), label="Original Trajectory")
        # plt.plot(time.cpu(), noisy_trajectory[0].cpu(), label="Noisy Trajectory")
        # plt.title("Sine Wave Trajectories")
        # plt.xlabel("Time")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.show()

        # Predict the error using the model
        predicted_bins = model(noisy_trajectory, random_timesteps).view(batch_size, sequence_length)

        # Compute the loss
        loss = F.mse_loss(predicted_bins, noise)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Sampling a new trajectory after training
def sample_trajectory(length=sequence_length, diffusion_steps=1000):
    sampled_trajectory = torch.randn(1, length, 1).to(device)

    scheduler.set_timesteps(diffusion_steps)

    for step in range(diffusion_steps):
        with torch.no_grad():
            t = scheduler.timesteps[step]

            # latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = model(sampled_trajectory, torch.tensor([t], device=device))

            # Normally we'd rely on the scheduler to handle the update step:
            sampled_trajectory = scheduler.step(noise_pred, t, sampled_trajectory).prev_sample

            # Instead, let's do it ourselves:
            # prev_t = max(1, t.item() - (scheduler.config.num_train_timesteps//diffusion_steps)) # t-1
            # alpha_t = scheduler.alphas_cumprod[t.item()]
            # alpha_t_prev = scheduler.alphas_cumprod[prev_t]
            # predicted_x0 = (sampled_trajectory - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
            # direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
            # sampled_trajectory = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

    # Plot the sampled trajectory
    plt.figure(figsize=(12, 6))
    plt.plot(time.cpu(), sampled_trajectory[0].cpu(), label="Sampled Trajectory")
    plt.title("Sampled Sine Wave Trajectory")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


for _ in range(20):
    # Plot the sampled trajectory
    sample_trajectory()
