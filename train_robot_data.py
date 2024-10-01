import torch
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

# Define the neural network architecture to model the denoising process.
class TrajectoryDenoisingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrajectoryDenoisingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.timestep_embedding = PositionalEncoding(32)

    def forward(self, x, timestep):
        # Flatten the input trajectories so we have one for each batched sample
        x = x.flatten(1)
        # Concatenate the timestep to each input trajectory
        x1 = torch.cat([x, self.timestep_embedding(timestep)], dim=1)
        return self.network(x1).view(-1, x.size(1))


# Positional embedding for timesteps
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_timestep=1000):
        super().__init__()
        self.dim = dim
        self.max_timestep = max_timestep

    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.unsqueeze(-1) * emb
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# Set up the diffusion scheduler (1000 timesteps as default)
num_train_timesteps = 1000
scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

# Define dimensions based on the sine wave generation
trajectory_dim = 1  # We only need 1D output for the sine wave
hidden_dim = 512

# Read the robot data from the CSV file
data = pd.read_csv('joint_commands.csv')

# Extract the joint command data for the LKnee joint
lknee_data = data[['LKnee']]

# Plot the LKnee joint data
plt.figure(figsize=(12, 6))
plt.plot(lknee_data)
plt.title("LKnee Joint Data")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()

# Normalize the LKnee joint data
lknee_data = (lknee_data - lknee_data.mean()) / (lknee_data.std())

# Drop every second data point to reduce the sequence length
lknee_data = lknee_data[::3]

# Chunk the data into sequences of n timesteps
timesteps = 70
time = torch.linspace(0, 1, timesteps).unsqueeze(-1)
real_trajectories = torch.tensor([lknee_data[i:i + timesteps].values for i in range(len(lknee_data) - timesteps)]).squeeze().float()

# Apply tanh to the normalized data to restrict the range
real_trajectories = torch.tanh(real_trajectories)


# Shuffle the data
torch.manual_seed(42)
real_trajectories = real_trajectories[torch.randperm(real_trajectories.size(0))]


# Plot
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(time, real_trajectories[i], label=f"Trajectory {i + 1}")
plt.title("LKnee Trajectories")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.show()

num_samples = real_trajectories.size(0)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move data to the GPU
real_trajectories = real_trajectories.to(device)
time = time.to(device)

# Initialize the model and optimizer
sequence_length = timesteps
model = TrajectoryDenoisingModel(input_dim=sequence_length, hidden_dim=hidden_dim, output_dim=sequence_length).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

batch_size = 500

# Training loop
for epoch in tqdm(range(500)):  # Number of training epochs
    for batch in range(num_samples // batch_size):
        optimizer.zero_grad()

        target = real_trajectories[batch*batch_size:(batch+1)*batch_size]

        # Sample a random timestep for each trajectory in the batch
        random_timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,)).long()

        # Sample noise to add to the entire trajectory
        noise = torch.randn_like(target).to(device)

        # Forward diffusion: Add noise to the entire trajectory at the random timestep
        noisy_trajectory = scheduler.add_noise(target, noise, random_timesteps)

        # Predict the denoised trajectory using the model
        pred = model(noisy_trajectory, random_timesteps.to(device))

        # Loss function: L2 loss between predicted and original trajectory
        loss = F.mse_loss(pred, noise)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Sampling a new sine wave trajectory after training
def sample_trajectory(steps=50):
    # Start with random noise as the input
    initial_noise = sampled_trajectory = torch.randn(1, timesteps).to(device)

    scheduler.set_timesteps(steps)

    with torch.no_grad():
        # Reverse diffusion: Gradually denoise the trajectory
        for t in range(steps - 1, -1, -1):
            predicted_noise = model(sampled_trajectory, torch.tensor([t], device=device))
            sampled_trajectory = scheduler.step(predicted_noise, t, sampled_trajectory).prev_sample

    return sampled_trajectory, initial_noise

while True:
    # Generate 5 new sine wave trajectories
    plt.figure(figsize=(12, 6))
    for i in range(1):
        sampled_trajectory, initial_noise = sample_trajectory()
        plt.plot(time.cpu(), sampled_trajectory.squeeze().detach().cpu(), label=f"Sampled Trajectory {i + 1}")
        plt.plot(time.cpu(), initial_noise.squeeze().detach().cpu(), label=f"Initial Noise")
    plt.title("Sampled Sine Wave Trajectories")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
