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
timesteps = 50
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

# Define the Transformer-based model
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, nhead=8, dropout=0.1):
        super(TransformerPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Input embedding layer to match input to hidden dimension
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer to project the transformer output back to joint space
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Positional encoding to inject time-related information
        self.positional_encoding = nn.Parameter(self.create_positional_encoding(timesteps, hidden_dim), requires_grad=False)

    def create_positional_encoding(self, length, d_model):
        # Create a sinusoidal positional encoding
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tgt):
        # Add positional encoding to input
        tgt_emb = self.input_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Masking to prevent attending to future positions
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

        # Dummy encoder output (not used)
        memory = torch.zeros(tgt.size(0), 1, self.hidden_dim).to(device)

        # Pass through the Transformer decoder
        output = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)

        # Project the output back to joint values
        output = self.fc_out(output)
        return output

# Initialize the model, loss function, and optimizer
input_dim = trajectory_dim  # Input is 12 joint values at each timestep
output_dim = trajectory_dim
hidden_dim = 64  # The size of the embeddings
num_layers = 3  # Number of transformer layers
nhead = 4  # Number of attention heads

model = TransformerPredictor(input_dim, hidden_dim, output_dim, num_layers=num_layers, nhead=nhead).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training settings
num_epochs = 50
sequence_length = timesteps
batch_size = 64

# Function for training the Transformer model
def train_model():
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0

        # Mini-batch training
        for i in tqdm(range(0, num_samples - batch_size, batch_size)):
            optimizer.zero_grad()

            # Get a batch of input sequences and target sequences
            target_batch = real_trajectories[i:i + batch_size]

            # Forward pass
            output = model(target_batch)

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

    # Start with the provided initial sequence
    predictions = start_sequence

    for _ in range(predict_length):

        # Use the most recent prediction as input
        with torch.no_grad():
            output = model(predictions[:, -sequence_length:, :])

        # Take the last timestep from the output
        new_timestep = output[:, -1:, :]
        predictions = torch.cat([predictions, new_timestep], dim=1)

    # Concatenate all predictions
    return predictions

# Test the model: generate a sequence of 200 timesteps starting from an initial sequence
start_sequence = real_trajectories[0, :50, :].unsqueeze(0)  # First 50 timesteps as the starting sequence
predicted_sequence = predict_autoregressively(start_sequence, predict_length=200)

# Plot the predicted trajectories
plt.figure(figsize=(12, 6))
for i in range(trajectory_dim):
    plt.subplot(3, 4, i + 1)
    plt.plot(predicted_sequence[0, :, i].cpu().numpy(), label="Predicted")
    plt.title(f"Joint {data.columns[i]}")
plt.suptitle("Predicted Joint Trajectories (Autoregressive)")
plt.show()
