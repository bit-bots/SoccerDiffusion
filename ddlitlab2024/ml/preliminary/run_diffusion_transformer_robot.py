import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ema_pytorch import EMA
from ddlitlab2024.ml.preliminary.train_diffusion_transformer_robot import TrajectoryTransformerModel

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define hyperparameters
hidden_dim = 256
num_layers = 4
num_heads = 4
sequence_length = 100
train_timesteps = 1000

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
trajectory_dim = len(joints)

# Initialize the Transformer model and optimizer, and move model to device
model = TrajectoryTransformerModel(
    num_joints=trajectory_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    max_seq_len=sequence_length,
).to(device)
ema = EMA(model, beta=0.9999)

scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
scheduler.config.num_train_timesteps = train_timesteps

# Load the model
ema.load_state_dict(torch.load("trajectory_transformer_model.pth"))


# Sampling a new trajectory after training
def sample_trajectory(length=sequence_length, step_size=100, diffusion_steps=15):
    scheduler.set_timesteps(diffusion_steps)

    context = torch.zeros(1, 0, trajectory_dim).to(device)

    for _ in range(length // step_size):
        sampled_trajectory = torch.randn(1, step_size, trajectory_dim).to(device)

        for t in scheduler.timesteps:
            with torch.no_grad():
                sample_trajectory_with_context = torch.cat([context, sampled_trajectory], dim=1)

                # Predict the noise residual
                noise_pred = ema(sample_trajectory_with_context, torch.tensor([t], device=device))[:, -step_size:]

                # Normally we'd rely on the scheduler to handle the update step:
                sampled_trajectory = scheduler.step(noise_pred, t, sampled_trajectory).prev_sample

            # Plot the context and the sampled trajectory
        context = torch.cat([context, sampled_trajectory], dim=1)

    # Undo the normalization
    context = context * model.std + model.mean

    # Plot the sampled trajectory
    plt.figure(figsize=(12, 6))
    for j in range(trajectory_dim):
        plt.subplot(3, 4, j + 1)
        plt.plot(context[0, :, j].cpu(), label="Sampled Trajectory")
        # Scale the y-axis to the range of the training data
        plt.ylim(-3.5, 3.5)
        plt.title(f"Joint {joints[j]}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


for _ in range(20):
    # Plot the sampled trajectory
    sample_trajectory()
