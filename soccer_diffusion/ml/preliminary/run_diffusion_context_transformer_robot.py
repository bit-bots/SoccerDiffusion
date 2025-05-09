import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ema_pytorch import EMA

from soccer_diffusion.ml.preliminary.train_diffusion_context_transformer_robot import TrajectoryTransformerModel

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define hyperparameters
hidden_dim = 256
num_layers = 4
num_heads = 4
sequence_length = 100
train_timesteps = 1000
action_context_length = 100
trajectory_prediction_length = 10

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
    max_action_context_length=action_context_length,
    trajectory_prediction_length=trajectory_prediction_length,
).to(device)
ema = EMA(model, beta=0.9999)

scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
scheduler.config["num_train_timesteps"] = train_timesteps

# Load the model
ema.load_state_dict(torch.load("trajectory_context_transformer_model.pth"))


# Sampling a new trajectory after training
def sample_trajectory(length=1000, diffusion_steps=15):
    scheduler.set_timesteps(diffusion_steps)

    context = torch.ones(1, action_context_length, trajectory_dim).to(device) * model.mean.view(1, 1, -1)

    for i in range(length // trajectory_prediction_length):
        sampled_trajectory = torch.randn(1, trajectory_prediction_length, trajectory_dim).to(device)

        plot_this = i > action_context_length // trajectory_prediction_length

        if plot_this:
            plt.figure(figsize=(12, 6))

        for t in scheduler.timesteps:
            with torch.no_grad():
                # Predict the noise residual
                noise_pred = ema(
                    context[:, -min(action_context_length, context.size(1)) :, :],
                    sampled_trajectory,
                    torch.tensor([t], device=device),
                )

                # Normally we'd rely on the scheduler to handle the update step:
                sampled_trajectory = scheduler.step(noise_pred, t, sampled_trajectory).prev_sample

                if plot_this:
                    # Plot the sampled trajectory
                    for j in range(trajectory_dim):
                        plt.subplot(3, 4, j + 1)
                        color = cm.viridis(t / scheduler.timesteps[0])
                        plt.plot(
                            torch.arange(trajectory_prediction_length) + context.size(1),
                            sampled_trajectory[0, :, j].cpu(),
                            label=f"Step {t}",
                            color=color,
                        )
                        # Scale the y-axis to the range of the training data
                        plt.title(f"Joint {joints[j]}")

        if plot_this:
            # Plot the context and the sampled trajectory
            for j in range(trajectory_dim):
                plt.subplot(3, 4, j + 1)
                plt.plot(context[0, :, j].cpu(), label="Context")
                plt.title(f"Joint {joints[j]}")

            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.show()

        context = torch.cat([context, sampled_trajectory], dim=1)

    # Undo the normalization
    context = context * model.std + model.mean

    # Plot the sampled trajectory
    plt.figure(figsize=(12, 6))
    for j in range(trajectory_dim):
        plt.subplot(3, 4, j + 1)
        plt.plot(context[0, 2 * action_context_length :, j].cpu(), label="Sampled Trajectory")
        # Scale the y-axis to the range of the training data
        # plt.ylim(-3.5, 3.5)
        plt.title(f"Joint {joints[j]}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


for _ in range(20):
    # Plot the sampled trajectory
    sample_trajectory()
