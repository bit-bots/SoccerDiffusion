import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from tqdm import tqdm
from matplotlib import cm
from ema_pytorch import EMA

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrajectoryTransformerModel(nn.Module):
    def __init__(self, num_joints, hidden_dim, num_layers, num_heads, max_seq_len):
        super().__init__()
        self.embedding = nn.Linear(num_joints, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len + 1)
        self.step_encoding = StepToken(hidden_dim, device=device)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                norm_first=True,
                activation="gelu"
            ),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(hidden_dim, num_joints)

    def forward(self, x, step):
        # x shape: (batch_size, seq_len, joint, num_bins)
        # Flatten the joint and bin dimensions into a single token dimension
        x = x.view(x.size(0), x.size(1), -1)
        # Embed the input
        x = self.embedding(x)
        # Positional encoding
        x += self.positional_encoding(x)
        # Add token for the step
        x = torch.cat([self.step_encoding(step), x], dim=1)
        # Create a causal mask
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        # Memory tensor (not used)
        memory = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)
        # Pass through the transformer decoder
        out = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)  # Causal mask applied
        # Remove the step token
        out = out[:, 1:]
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
class StepToken(nn.Module):
    def __init__(self, dim, device=device):
        super().__init__()
        self.dim = dim
        self.token = nn.Parameter(torch.randn(1, dim // 2, device=device))

    def forward(self, x):
        half_dim = self.dim // 4
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -np.log(10000) / (half_dim - 1))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos(), self.token.expand((x.size(0), self.dim // 2))), dim=-1).unsqueeze(1)
        return emb


# Define dimensions for the Transformer model
trajectory_dim = 1  # 1D input for the sine wave
hidden_dim = 256
num_layers = 4
num_heads = 4
sequence_length = 30

# Generate a dataset of sine wave trajectories (500 samples)
num_samples = 5000
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
ema = EMA(model, beta=0.9999)

lr = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * (num_samples // batch_size))

scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2")
scheduler.config.num_train_timesteps = 1000

# Training loop
for epoch in tqdm(range(epochs)):  # Number of training epochs
    mean_loss = 0
    for batch in range(num_samples // batch_size):
        targets = real_trajectories[batch * batch_size : (batch + 1) * batch_size].to(device)

        optimizer.zero_grad()

        # Sample a random timestep for each trajectory in the batch
        random_timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,)).long().to(device)

        # Sample noise to add to the entire trajectory
        noise = torch.randn_like(targets).to(device)

        # Forward diffusion: Add noise to the entire trajectory at the random timestep
        noisy_trajectory = scheduler.add_noise(targets, noise, random_timesteps)

        # Predict the error using the model
        predicted_bins = model(noisy_trajectory, random_timesteps).view(batch_size, sequence_length)

        # Compute the loss
        loss = F.mse_loss(predicted_bins, noise)

        mean_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.update()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {mean_loss / (num_samples // batch_size)}, LR: {lr_scheduler.get_last_lr()[0]}")


# Sampling a new trajectory after training
def sample_trajectory(length=sequence_length, step_size=30, diffusion_steps=50):
    scheduler.set_timesteps(diffusion_steps)

    context = torch.zeros(1, 0, 1).to(device)

    for _ in range(length // step_size):

        sampled_trajectory = torch.randn(1, step_size, 1).to(device)

        plt.figure(figsize=(12, 6))

        for t in scheduler.timesteps:
            with torch.no_grad():

                sample_trajectory_with_context = torch.cat([context, sampled_trajectory], dim=1)

                # Predict the noise residual
                noise_pred = ema(sample_trajectory_with_context, torch.tensor([t], device=device))[:, -step_size:]

                # Normally we'd rely on the scheduler to handle the update step:
                sampled_trajectory = scheduler.step(noise_pred, t, sampled_trajectory).prev_sample

            # Plot the context and the sampled trajectory
            color = cm.viridis(t / scheduler.config.num_train_timesteps)
            plt.plot(time.cpu()[context.size(1) : context.size(1) + step_size, 0], sampled_trajectory[0,:,0].cpu(), label="Sampled Trajectory", color=color)
        plt.plot(time.cpu()[: context.size(1), 0], context[0,:,0].cpu(), label="Context", color=color)
        if context.size(1) > 0:
            plt.plot(time.cpu()[context.size(1) - 1: context.size(1) + 1], [context[0, -1, 0].cpu(), sampled_trajectory[0, 0, 0].cpu()], color='black')
        plt.title("Sampled Sine Wave Trajectory")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

        context = torch.cat([context, sampled_trajectory], dim=1)

    # Plot the sampled trajectory
    plt.figure(figsize=(12, 6))
    plt.plot(time.cpu(), context[0].cpu(), label="Sampled Trajectory")
    plt.title("Sampled Sine Wave Trajectory")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


for _ in range(20):
    # Plot the sampled trajectory
    sample_trajectory()

# Save the model
torch.save(ema.state_dict(), "trajectory_transformer_model.pth")