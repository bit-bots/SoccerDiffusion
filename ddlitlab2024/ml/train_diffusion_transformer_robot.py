import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ema_pytorch import EMA
from torch import nn
from tqdm import tqdm
import pandas as pd

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
                activation="gelu",
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
        # Memory tensor (not used)
        memory = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)
        # Pass through the transformer decoder
        out = self.transformer_decoder(x, memory)  # Causal mask applied
        # Remove the step token
        out = out[:, 1:]
        # Final classification layer (logits for each bin)
        return self.fc_out(out)


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


# Define hyperparameters
hidden_dim = 256
num_layers = 4
num_heads = 4
sequence_length = 100
epochs = 40
batch_size = 64
lr = 1e-4
train_timesteps = 1000

# Read the robot data from the CSV file
data = pd.read_csv("joint_commands.csv")

# Extract the joint command data all joints, and drop the time column
joints = [
    #"LHipYaw",
    #"RHipYaw",
    #"LHipRoll",
    #"RHipRoll",
    #"LHipPitch",
    #"RHipPitch",
    "LKnee",
    "RKnee",
    #"LAnklePitch",
    #"RAnklePitch",
    #"LAnkleRoll",
    #"RAnkleRoll",
]
data = data[joints]
trajectory_dim = len(joints)

# Drop every second data point to reduce the sequence length (subsample) TODO proper subsampling
data = data[::3]

# Normalize the joint data (-pi to pi) to (-1, 1)
data = (data - data.min()) / (data.max() - data.min()) * 2 - 1

# Chunk the data into sequences of 50 timesteps
timesteps = sequence_length
time = torch.linspace(0, 1, timesteps).unsqueeze(-1)
real_trajectories = (
    torch.tensor(np.array([data[i : i + timesteps].values for i in range(len(data) - timesteps)])).squeeze().float()
)

num_samples = real_trajectories.size(0)
real_trajectories = real_trajectories[torch.randperm(real_trajectories.size(0))]

# Subplot each joint, showing the first n batches
n = 1
plt.figure(figsize=(12, 6))
for i in range(trajectory_dim):
    plt.subplot(3, 4, i + 1)
    plt.plot(real_trajectories[:n, :, i].T)
    plt.title(f"Joint {data.columns[i]}")
plt.suptitle("LKnee Trajectories")
plt.show()

# Initialize the Transformer model and optimizer, and move model to device
model = TrajectoryTransformerModel(
    num_joints=trajectory_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    max_seq_len=sequence_length,
).to(device)
ema = EMA(model, beta=0.9999)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, total_steps=epochs * (num_samples // batch_size)
)

scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2")
scheduler.config.num_train_timesteps = train_timesteps

# Training loop
for epoch in range(epochs):  # Number of training epochs
    mean_loss = 0
    for batch in tqdm(range(num_samples // batch_size)):
        targets = real_trajectories[batch * batch_size : (batch + 1) * batch_size].to(device)

        optimizer.zero_grad()

        # Sample a random timestep for each trajectory in the batch
        random_timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,)).long().to(device)

        # Sample noise to add to the entire trajectory
        noise = torch.randn_like(targets).to(device)

        # Forward diffusion: Add noise to the entire trajectory at the random timestep
        noisy_trajectory = scheduler.add_noise(targets, noise, random_timesteps)

        # Predict the error using the model
        predicted_traj = model(noisy_trajectory, random_timesteps)

        # Compute the loss
        loss = F.mse_loss(predicted_traj, noise)

        mean_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.update()

        if (batch + 1) % 100 == 0:
            print(f"Epoch {epoch}, Loss: {mean_loss / batch}, LR: {lr_scheduler.get_last_lr()[0]}")


# Save the model
torch.save(ema.state_dict(), "trajectory_transformer_model.pth")
