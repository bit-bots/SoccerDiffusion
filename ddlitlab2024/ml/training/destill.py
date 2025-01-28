from dataclasses import asdict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddlitlab2024.dataset.pytorch import DDLITLab2024Dataset, Normalizer, worker_init_fn
from ddlitlab2024.ml import logger
from ddlitlab2024.ml.model import End2EndDiffusionTransformer
from ddlitlab2024.ml.model.encoder.image import ImageEncoderType, SequenceEncoderType
from ddlitlab2024.ml.model.encoder.imu import IMUEncoder

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix tqdm when the terminal width changes, this is for some reason not a default, therfore we make it one
tqdm = partial(tqdm, dynamic_ncols=True)

if __name__ == "__main__":
    logger.info("Starting training")
    logger.info(f"Using device {device}")
    # TODO wandb
    # Define hyperparameters # TODO proper configuration
    hidden_dim = 256
    num_layers = 4
    num_heads = 4
    action_context_length = 100
    trajectory_prediction_length = 10
    epochs = 50
    batch_size = 16
    lr = 1e-4
    train_denoising_timesteps = 1000
    image_context_length = 10
    action_context_length = 100
    imu_context_length = 100
    joint_state_context_length = 100
    num_normalization_samples = 1000
    inference_denosing_timesteps = 30
    num_joints = 20
    checkpoint: str = "trajectory_transformer_model.pth"

    # Load the dataset (primary for example conditioning)
    logger.info("Create dataset objects")
    dataset = DDLITLab2024Dataset(
        num_joints=num_joints,
        num_frames_video=image_context_length,
        num_samples_joint_trajectory_future=trajectory_prediction_length,
        num_samples_joint_trajectory=action_context_length,
        num_samples_imu=imu_context_length,
        num_samples_joint_states=joint_state_context_length,
    )
    num_workers = 5
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DDLITLab2024Dataset.collate_fn,
        persistent_workers=num_workers > 1,
        # prefetch_factor=10 * num_workers,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    # Initialize the Transformer model and optimizer, and move model to device
    teacher_model = End2EndDiffusionTransformer(  # TODO enforce all params to be consistent with the dataset
        num_joints=num_joints,
        hidden_dim=hidden_dim,
        use_action_history=True,
        num_action_history_encoder_layers=2,
        max_action_context_length=action_context_length,
        use_imu=True,
        imu_orientation_embedding_method=IMUEncoder.OrientationEmbeddingMethod.QUATERNION,
        num_imu_encoder_layers=2,
        max_imu_context_length=imu_context_length,
        use_joint_states=True,
        joint_state_encoder_layers=2,
        max_joint_state_context_length=joint_state_context_length,
        use_images=True,
        image_sequence_encoder_type=SequenceEncoderType.TRANSFORMER,
        image_encoder_type=ImageEncoderType.RESNET18,
        num_image_sequence_encoder_layers=1,
        max_image_context_length=image_context_length,
        num_decoder_layers=4,
        trajectory_prediction_length=trajectory_prediction_length,
    ).to(device)

    # Utilize an Exponential Moving Average (EMA) for the model to smooth out the training process
    teacher_ema = EMA(teacher_model, beta=0.999)

    # Load the model if a checkpoint is provided
    if checkpoint is not None:
        logger.info(f"Loading model from {checkpoint}")
        teacher_ema.load_state_dict(torch.load(checkpoint, weights_only=True))

    # Clone the model
    student_model = teacher_model.clone()
    student_ema = EMA(student_model, beta=0.999)

    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(dataloader))

    # Create diffusion noise scheduler
    scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.config["num_train_timesteps"] = train_denoising_timesteps

    # Training loop
    for epoch in range(epochs):
        mean_loss = 0

        # Iterate over the dataset
        for i, batch in enumerate(pbar := tqdm(dataloader)):
            # Move the data to the device
            batch = {k: v.to(device) for k, v in asdict(batch).items()}

            # Extract the target actions
            joint_targets = batch["joint_command"]

            # Sample our initial trajectory to denoise
            noisy_trajectory = torch.randn_like(joint_targets).to(device)

            # Reset the gradients
            optimizer.zero_grad()

            with torch.no_grad():
                # Clone the noisy trajectory because we want to keep the original as input for the student model
                trajectory = noisy_trajectory.clone()

                # Perform the embedding of the conditioning
                embedded_input = teacher_model.encode_input_data(batch)

                scheduler.set_timesteps(inference_denosing_timesteps)
                for t in scheduler.timesteps:
                    with torch.no_grad():
                        # Predict the noise residual
                        noise_pred = teacher_model.forward_with_context(
                            embedded_input, trajectory, torch.tensor([t], device=device)
                        )

                        # Update the trajectory based on the predicted noise and the current step of the denoising process
                        trajectory = scheduler.step(noise_pred, t, trajectory).prev_sample

            # Predict the denoised trajectory directly using the student model (null the timestep, as we are doing a single step prediction)
            student_trajectory_prediction = student_model(batch, noisy_trajectory, torch.zeros(joint_targets.size(0), device=device))

            # Compute the loss
            loss = F.mse_loss(student_trajectory_prediction, trajectory)

            mean_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            student_ema.update()

            pbar.set_postfix_str(
                f"Epoch {epoch}, Loss: {mean_loss / (i + 1):.05f}, LR: {lr_scheduler.get_last_lr()[0]:0.7f}"
            )

        # Save the model
        torch.save(student_ema.state_dict(), "destilled_trajectory_transformer_model.pth")
