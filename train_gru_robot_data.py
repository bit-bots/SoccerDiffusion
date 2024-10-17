import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd


# Define dimensions based on the sine wave generation
trajectory_dim = 12  # We have two legs with 6 joints each
hidden_dim = 255

# Read the robot data from the CSV file
data = pd.read_csv('joint_commands.csv')

# Extract the joint command data all joints, and drop the time column
data = data[["LHipYaw", "LHipRoll","LHipPitch","LKnee","LAnklePitch","LAnkleRoll","RHipYaw","RHipRoll","RHipPitch","RKnee","RAnklePitch","RAnkleRoll"]]

# Plot the LKnee joint data
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("LKnee Joint Data")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()

# Drop every second data point to reduce the sequence length (subsample) TODO proper subsampling
data = data[::3]

# Chunk the data into sequences of 50 timesteps
timesteps = 7
time = torch.linspace(0, 1, timesteps).unsqueeze(-1)
real_trajectories = torch.tensor(np.array([data[i:i + timesteps].values for i in range(len(data) - timesteps)])).squeeze().float()

# Apply tanh to the normalized data to restrict the range
# real_trajectories = torch.tanh(real_trajectories)

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


# Continue importing necessary modules
import torch.optim as optim

# Define the GRU-based model
class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GRUPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer (to map the hidden state to the joint output)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        # Pass the input through the GRU layer
        out, hidden = self.gru(x, hidden)

        # Map the GRU outputs to joint values (output_dim = 12 joint values)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

# Initialize model, loss function, and optimizer
input_dim = trajectory_dim  # Input is 12 joint values at each timestep
output_dim = trajectory_dim
model = GRUPredictor(input_dim, hidden_dim, output_dim).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training settings
num_epochs = 10
sequence_length = timesteps
batch_size = 64

# Function for training the GRU model
def train_model():
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        hidden = model.init_hidden(batch_size)

        # Mini-batch training
        for i in range(0, num_samples - batch_size, batch_size):
            optimizer.zero_grad()

            # Get a batch of input sequences and target sequences
            input_batch = real_trajectories[i:i + batch_size, :-1, :]
            target_batch = real_trajectories[i:i + batch_size, 1:, :]

            # Forward pass
            hidden = hidden.detach()  # Detach the hidden state to avoid backprop through the entire history
            output, hidden = model(input_batch, hidden)

            # Compute loss
            loss = loss_fn(output, target_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Call the training function
train_model()

# Autoregressive prediction function
def predict_autoregressively(start_sequence, predict_length):
    model.eval()

    # Initialize hidden state
    hidden = model.init_hidden(1)

    # Start with the provided initial sequence
    predictions = [start_sequence]

    # Predict step-by-step
    current_input = start_sequence.unsqueeze(0)

    for _ in range(predict_length):
        # Predict the next joint positions
        with torch.no_grad():
            print(current_input.shape)
            output, hidden = model(current_input, hidden)
            print(output.shape)

        # Append the output to the predictions list
        predictions.append(output.squeeze(0))

        # Use the output as the next input
        current_input = output

    # Concatenate all predictions
    return torch.cat(predictions, dim=0)

# Test the model: generate a sequence of 200 timesteps starting from an initial sequence
start_sequence = real_trajectories[0, :50, :]  # First 50 timesteps as the starting sequence
print(start_sequence.shape)
predicted_sequence = predict_autoregressively(start_sequence, predict_length=3)
print(predicted_sequence.shape)

# Plot the predicted trajectories
plt.figure(figsize=(12, 6))
for i in range(trajectory_dim):
    plt.subplot(3, 4, i + 1)
    plt.plot(predicted_sequence[:, i].cpu().numpy(), label="Predicted")
    plt.title(f"Joint {data.columns[i]}")
plt.suptitle("Predicted Joint Trajectories (Autoregressive)")
plt.show()
