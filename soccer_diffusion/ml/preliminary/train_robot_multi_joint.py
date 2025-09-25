import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn
from tqdm import tqdm


# Define the neural network architecture to model the denoising process.
class TrajectoryDenoisingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.joint_enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU())
        self.joint_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, output_dim)
        )
        self.timestep_embedding = PositionalEncoding(hidden_dim)

    def forward(self, x, timestep):
        # Flatten the input trajectories so we have one for each batched sample
        x = x.flatten(1)
        x = self.joint_enc(x) + self.timestep_embedding(timestep)
        x = self.joint_dec(x)
        return x


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
trajectory_dim = 12  # We have two legs with 6 joints each
hidden_dim = 2048 * 2

# Read the robot data from the CSV file
data = pd.read_csv("joint_commands.csv")

# Extract the joint command data all joints, and drop the time column
data = data[
    [
        "LHipYaw",
        "LHipRoll",
        "LHipPitch",
        "LKnee",
        "LAnklePitch",
        "LAnkleRoll",
        "RHipYaw",
        "RHipRoll",
        "RHipPitch",
        "RKnee",
        "RAnklePitch",
        "RAnkleRoll",
    ]
]

# Plot the LKnee joint data
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("LKnee Joint Data")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()

# Normalize the LKnee joint data (per joint)
data = (data - data.mean()) / data.std()

# Drop every second data point to reduce the sequence length (subsample) TODO proper subsampling
data = data[::3]

# Chunk the data into sequences of 50 timesteps
timesteps = 70
time = torch.linspace(0, 1, timesteps).unsqueeze(-1)
real_trajectories = (
    torch.tensor(np.array([data[i : i + timesteps].values for i in range(len(data) - timesteps)])).squeeze().float()
)

# Apply tanh to the normalized data to restrict the range
real_trajectories = torch.tanh(real_trajectories)

# Shuffle the data
torch.manual_seed(42)
real_trajectories = real_trajectories[torch.randperm(real_trajectories.size(0))]


# Subplot each joint, showing the first 5 batches
plt.figure(figsize=(12, 6))
for i in range(trajectory_dim):
    plt.subplot(3, 4, i + 1)
    plt.plot(real_trajectories[:5, :, i].T)
    plt.title(f"Joint {data.columns[i]}")
plt.suptitle("LKnee Trajectories")
plt.show()

num_samples = real_trajectories.size(0)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move data to the GPU
real_trajectories = real_trajectories.to(device)
time = time.to(device)

batch_size = 64
epochs = 200
lr = 1e-4

# Initialize the model and optimizer
sequence_length = timesteps
model = TrajectoryDenoisingModel(
    input_dim=sequence_length * trajectory_dim, hidden_dim=hidden_dim, output_dim=sequence_length * trajectory_dim
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, steps_per_epoch=num_samples // batch_size, epochs=epochs
)

# Training loop
for epoch in tqdm(range(epochs)):  # Number of training epochs
    for batch in range(num_samples // batch_size):
        optimizer.zero_grad()

        target = real_trajectories[batch * batch_size : (batch + 1) * batch_size].view(batch_size, -1)

        # Sample a random timestep for each trajectory in the batch
        random_timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,)).long()

        # Sample noise to add to the entire trajectory
        noise = torch.randn_like(target, device=device)

        # Forward diffusion: Add noise to the entire trajectory at the random timestep
        noisy_trajectory = scheduler.add_noise(target, noise, random_timesteps)

        # Predict the denoised trajectory using the model
        pred = model(noisy_trajectory, random_timesteps.to(device))

        # Loss function: L2 loss between predicted and original trajectory
        loss = F.mse_loss(pred, noise)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Sampling a new sine wave trajectory after training
def sample_trajectory(steps=50):
    # Start with random noise as the input
    initial_noise = sampled_trajectory = torch.randn(1, timesteps * trajectory_dim, device=device)

    scheduler.set_timesteps(steps)

    with torch.no_grad():
        # Reverse diffusion: Gradually denoise the trajectory
        for t in range(steps - 1, -1, -1):
            predicted_noise = model(sampled_trajectory, torch.tensor([t], device=device))
            sampled_trajectory = scheduler.step(predicted_noise, t, sampled_trajectory).prev_sample

    return sampled_trajectory.view(1, sequence_length, trajectory_dim), initial_noise.view(
        1, sequence_length, trajectory_dim
    )


while True:
    # Generate 5 new sine wave trajectories
    plt.figure(figsize=(12, 6))
    sampled_trajectory, initial_noise = sample_trajectory()
    # plot the sampled trajectory for each joint in a subplot
    for j in range(trajectory_dim):
        plt.subplot(3, 4, j + 1)
        plt.plot(sampled_trajectory[0, :, j].cpu().numpy(), label="Sampled Trajectory")
        # plt.plot(initial_noise[0, :, j].cpu().numpy(), label="Initial Noise")
        plt.title(f"Joint {data.columns[j]}")

    # plt.title("Sampled Sine Wave Trajectories")
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
