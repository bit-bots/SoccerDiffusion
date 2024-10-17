import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
from tqdm import tqdm
import pandas as pd

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Transformer Decoder model for velocity prediction
class TrajectoryTransformerModel(nn.Module):
    def __init__(self, num_joints, hidden_dim, num_layers, num_heads, max_seq_len, num_bins):
        super(TrajectoryTransformerModel, self).__init__()
        self.embedding = nn.Linear(num_joints*num_bins, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_dim, num_joints*num_bins)

    def forward(self, x):
        # x shape: (batch_size, seq_len, joint, num_bins)
        # Flatten the joint and bin dimensions into a single token dimension
        x = x.view(x.size(0), x.size(1), -1)
        # Positional encoding
        x = self.embedding(x) + self.positional_encoding(x)
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
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

# Positional Encoding class for the Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x.device)

# Define dimensions for the Transformer model
hidden_dim = 128
num_layers = 1
num_heads = 4
num_bins = 512
sequence_length = 30

# Read the robot data from the CSV file
data = pd.read_csv('joint_commands.csv')

# Extract the joint command data all joints, and drop the time column
joints = ["LKnee","RKnee"]
data = data[joints]
trajectory_dim = len(joints)

# Plot the LKnee joint data
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("LKnee Joint Data")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()

print(data.min(), data.max())

# Drop every second data point to reduce the sequence length (subsample) TODO proper subsampling
data = data[::6]

# Chunk the data into sequences of 50 timesteps
timesteps = sequence_length
time = torch.linspace(0, 1, timesteps).unsqueeze(-1)
real_trajectories = torch.tensor(np.array([data[i:i + timesteps].values for i in range(len(data) - timesteps)])).squeeze().float()

num_samples = real_trajectories.size(0)
real_trajectories = real_trajectories[torch.randperm(real_trajectories.size(0))]

# Compute velocities (finite differences) and discretize them into 30 bins
def compute_velocity(trajectories):
    velocities = torch.diff(trajectories, dim=1)
    return velocities

def discretize_velocity(velocities, num_bins):
    bin_edges = torch.linspace(-0.25, 0.25, num_bins + 1).to(device)
    binned_velocities = torch.bucketize(velocities, bin_edges)
    return binned_velocities, bin_edges

# Plot the first 5 sine wave trajectories
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(time.cpu(), real_trajectories[i].cpu(), label=f"Trajectory {i + 1}")
plt.title("Sine Wave Trajectories")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Initialize the Transformer model and optimizer, and move model to device
model = TrajectoryTransformerModel(num_joints=trajectory_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                   num_heads=num_heads, max_seq_len=sequence_length, num_bins=num_bins).to(device)
summary(model, input_size=(1, 30, trajectory_dim, num_bins))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create batches of data
batch_size = 32

# Training loop
for epoch in tqdm(range(10)):  # Number of training epochs
    for batch in range(num_samples // batch_size):
        targets = real_trajectories[batch * batch_size: (batch + 1) * batch_size].to(device)

        optimizer.zero_grad()

        # Map -1 pi - pi to 0 - 1
        targets_scaled = (targets + np.pi) / (2 * np.pi)

        # Discretize into num_bins
        targets_binned = (targets_scaled * num_bins).long()

        # Make floating point tensors
        targets_binned = targets_binned.unsqueeze(-1).to(device)

        # Make one-hot encoding of the binned velocities
        binned_velocities_embedding = F.one_hot(targets_binned, num_bins).float().to(device)

        # Cut and shift the input sequence for the Transformer model
        input_seq = binned_velocities_embedding[:, :-1]
        target_seq = targets_binned[:, 1:]

        # Predict the velocity bins using the Transformer model
        predicted_bins = model(input_seq)

        # Cross-entropy loss between predicted bins and actual velocity bins
        loss = F.cross_entropy(predicted_bins.reshape(-1, num_bins), target_seq.reshape(-1))

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Sampling a new trajectory after training
def sample_trajectory(steps=20):
    sampled_trajectory = torch.zeros(1, 1, trajectory_dim, num_bins).to(device) + F.one_hot(torch.randint(0, num_bins, (1, 1)), num_bins).float().to(device)

    probabilities = []

    top_classes = []

    for _ in range(steps):
        # Predict the next velocity bin using the Transformer model
        predicted_bin = model(sampled_trajectory.view(1, -1, trajectory_dim * num_bins)).view(1, -1, trajectory_dim, num_bins)

        probabilities.append(predicted_bin[:, -1, 0].squeeze(0).cpu().detach().numpy())

        # Sample top bin as the next velocity
        _, sampled_bin = torch.topk(predicted_bin, k=1, dim=-1)

        # Only keep the last predicted bin
        sampled_bin = sampled_bin[:, -1]

        top_classes.append(sampled_bin.tolist())
        print(top_classes)

        # One-hot encode the sampled bin
        sampled_bin_onehot = F.one_hot(sampled_bin, num_bins).float().to(device).squeeze(2).unsqueeze(0)

        # Append the sampled bin to the trajectory
        sampled_trajectory = torch.cat([sampled_trajectory, sampled_bin_onehot], dim=1)

    # Plot heatmap of the predicted bins
    plt.figure(figsize=(12, 6))
    plt.imshow(np.asarray(probabilities).T, cmap='hot', interpolation='nearest', aspect='auto')
    plt.title("Predicted Bins")
    plt.xlabel("Time")
    plt.ylabel("Bin")
    plt.show()
    return top_classes


for _ in range(20):
    # Plot the sampled trajectory
    sampled_trajectory = sample_trajectory(steps=30)
    # Coverting the sampled trajectory to a numpy array
    sampled_trajectory = np.array(sampled_trajectory)
    # Convert back to radians
    sampled_trajectory = (sampled_trajectory / num_bins) * (2 * np.pi) - np.pi
    plt.figure(figsize=(12, 6))
    # plot the sampled trajectory for each joint in a subplot
    for j in range(trajectory_dim):
        plt.subplot(1, 2, j + 1)
        plt.plot(sampled_trajectory[:, 0, j, 0], label="Sampled Trajectory")
        # Fix limits to -pi to pi
        plt.ylim(-np.pi, np.pi)
        plt.title(f"Joint {joints[j]}")
    plt.legend()
    plt.show()
