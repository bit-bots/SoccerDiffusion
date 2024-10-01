import torch
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        emb = self.timestep_embedding(timestep)
        x1 = torch.cat([x, emb], dim=1)
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
scheduler = DDIMScheduler(num_train_timesteps=1000)

# Define dimensions based on the sine wave generation
trajectory_dim = 1  # We only need 1D output for the sine wave
hidden_dim = 255

# Generate a dataset of sine wave trajectories (1000 samples, shifted by random phase)
sequence_length = 30
num_samples = 500
time = torch.linspace(0, 2 * np.pi, sequence_length).unsqueeze(-1)
real_trajectories = torch.sin(time + torch.rand(1, num_samples) * 2 * np.pi).permute(1, 0)


# Plot the first 5 sine wave trajectories
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(time, real_trajectories[i], label=f"Trajectory {i + 1}")
plt.title("Sine Wave Trajectories")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Initialize the model and optimizer
model = TrajectoryDenoisingModel(input_dim=sequence_length, hidden_dim=hidden_dim, output_dim=sequence_length)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in tqdm(range(10000)):  # Number of training epochs
    optimizer.zero_grad()

    # Sample a random timestep for each trajectory in the batch
    random_timesteps = torch.randint(0, scheduler.num_train_timesteps, (num_samples,), device=real_trajectories.device).long()

    # Sample noise to add to the entire trajectory
    noise = torch.randn_like(real_trajectories)

    # Forward diffusion: Add noise to the entire trajectory at the random timestep
    noisy_trajectory = scheduler.add_noise(real_trajectories, noise, random_timesteps)

    # Predict the denoised trajectory using the model
    predicted_noise = model(noisy_trajectory, random_timesteps).view(num_samples, sequence_length)

    # Plot predicted noise vs original noise\
    if epoch % 100 == 0 and False:
        plt.figure(figsize=(12, 6))
        plt.plot(time, noise[0], label="Original Noise")
        plt.plot(time, predicted_noise[0].detach(), label="Predicted Noise")
        plt.title("Original vs Predicted Noise")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    # Loss function: L2 loss between predicted and original trajectory
    loss = F.mse_loss(predicted_noise, noise)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Sampling a new sine wave trajectory after training
def sample_trajectory(steps=50):
    # Start with random noise as the input
    sampled_trajectory = torch.randn(1, sequence_length)

    traj_hist = []

    scheduler.set_timesteps(steps)

    # Reverse diffusion: Gradually denoise the trajectory
    for t in range(steps - 1, -1, -1):
        predicted_noise = model(sampled_trajectory, torch.tensor([t]))
        sampled_trajectory = scheduler.step(predicted_noise, t, sampled_trajectory).prev_sample
        traj_hist.append(sampled_trajectory)


    #plt.figure(figsize=(12, 6))
    #plt.plot(time, init_noise.squeeze().detach(), label=f"Initial Noise")
    #for i, t in enumerate(traj_hist[::steps//10]):
    #    plt.plot(time, t.squeeze().detach(), label=f"Sampled Trajectory {i}")
    #plt.plot(time, sampled_trajectory.squeeze().detach(), label=f"Sampled Trajectory")
    #plt.title("Sampled Sine Wave Trajectories")
    #plt.xlabel("Time")
    #plt.ylabel("Amplitude")
    #plt.legend()
    #plt.show()

    return sampled_trajectory

# Generate 5 new sine wave trajectories
plt.figure(figsize=(12, 6))
for i in range(5):
    sampled_trajectory = sample_trajectory()
    plt.plot(time, sampled_trajectory.squeeze().detach(), label=f"Sampled Trajectory {i + 1}")
plt.title("Sampled Sine Wave Trajectories")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()