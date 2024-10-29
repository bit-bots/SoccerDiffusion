import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ema_pytorch import EMA
from tqdm import tqdm

from ddlitlab2024.ml.model import End2EndDiffusionTransformer

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # TODO wandb
    # Define hyperparameters # TODO proper configuration
    hidden_dim = 256
    num_layers = 4
    num_heads = 4
    action_context_length = 100
    trajectory_prediction_length = 10
    epochs = 400
    batch_size = 128
    lr = 1e-4
    train_timesteps = 1000

    # Read the robot data from the CSV file # TODO proper data loading
    data = pd.read_csv("joint_commands.csv")

    # Extract the joint command data all joints, and drop the time column
    joints = [
        "LHipYaw",
        "RHipYaw",
        "LHipRoll",
        "RHipRoll",
        "LHipPitch",
        "RHipPitch",
        "LKnee",
        "RKnee",
        "LAnklePitch",
        "RAnklePitch",
        "LAnkleRoll",
        "RAnkleRoll",
    ]
    data = data[joints]
    trajectory_dim = len(joints)

    # Drop every second data point to reduce the sequence length (subsample) TODO proper subsampling
    data = data[::3]

    # Normalize the joint data
    stds = data.std()
    means = data.mean()
    data = (data - means) / stds

    # Chunk the data into sequences of 50 timesteps
    timesteps = action_context_length + trajectory_prediction_length
    time = torch.linspace(0, 1, timesteps).unsqueeze(-1)
    real_trajectories = (
        torch.tensor(np.array([data[i : i + timesteps].values for i in range(len(data) - timesteps)])).squeeze().float()
    )

    num_samples = real_trajectories.size(0)

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
    model = End2EndDiffusionTransformer(
        num_joints=trajectory_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_action_context_length=action_context_length,
        trajectory_prediction_length=trajectory_prediction_length,
    ).to(device)

    # Add normalization parameters to the model
    model.mean = torch.tensor(means.values).to(device)
    model.std = torch.tensor(stds.values).to(device)

    ema = EMA(model, beta=0.9999)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * (num_samples // batch_size)
    )

    scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.config.num_train_timesteps = train_timesteps

    # Training loop
    for epoch in range(epochs):  # Number of training epochs
        mean_loss = 0
        # Shuffle the data for each epoch
        real_trajectories = real_trajectories[torch.randperm(real_trajectories.size(0))]

        for batch in tqdm(range(num_samples // batch_size)):
            targets = real_trajectories[batch * batch_size : (batch + 1) * batch_size].to(device)

            # Split the data into past actions and noisy action predictions
            past_actions = targets[:, :action_context_length]
            target_actions = targets[:, action_context_length:]

            optimizer.zero_grad()

            # Sample a random timestep for each trajectory in the batch
            random_timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,)).long().to(device)

            # Sample noise to add to the entire trajectory
            noise = torch.randn_like(target_actions).to(device)

            # Forward diffusion: Add noise to the entire trajectory at the random timestep
            noisy_trajectory = scheduler.add_noise(target_actions, noise, random_timesteps)

            # Predict the error using the model
            predicted_traj = model(past_actions, noisy_trajectory, random_timesteps)

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
