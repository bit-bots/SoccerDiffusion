import argparse
import os
import sys
import torch
from ema_pytorch import EMA
import yaml

from ddlitlab2024.ml.model import End2EndDiffusionTransformer
from ddlitlab2024.ml.model.encoder.image import ImageEncoderType, SequenceEncoderType
from ddlitlab2024.ml.model.encoder.imu import IMUEncoder

# This script embeds the parameters into the model itself

if __name__ == "__main__":

    # Get command line arguments
    parser = argparse.ArgumentParser(description="Convert a legacy checkpoint to the new format")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("output", type=str, help="Path to save the model")
    args = parser.parse_args()

    # Load the hyperparameters from training yaml file
    with open(args.config, "r") as file:
        params = yaml.safe_load(file)

    # Initialize the Transformer model and optimizer, and move model to device
    model = End2EndDiffusionTransformer(
        num_joints=params["num_joints"],
        hidden_dim=params["hidden_dim"],
        use_action_history=params["use_action_history"],
        num_action_history_encoder_layers=params["num_action_history_encoder_layers"],
        max_action_context_length=params["action_context_length"],
        use_imu=params["use_imu"],
        imu_orientation_embedding_method=IMUEncoder.OrientationEmbeddingMethod(params["imu_orientation_embedding_method"]),
        num_imu_encoder_layers=params["num_imu_encoder_layers"],
        imu_context_length=params["imu_context_length"],
        use_joint_states=params["use_joint_states"],
        joint_state_encoder_layers=params["joint_state_encoder_layers"],
        joint_state_context_length=params["joint_state_context_length"],
        use_images=params["use_images"],
        image_sequence_encoder_type=SequenceEncoderType(params["image_sequence_encoder_type"]),
        image_encoder_type=ImageEncoderType(params["image_encoder_type"]),
        num_image_sequence_encoder_layers=params["num_image_sequence_encoder_layers"],
        image_context_length=params["image_context_length"],
        num_decoder_layers=params["num_decoder_layers"],
        trajectory_prediction_length=params["trajectory_prediction_length"],
    )

    ema_model = EMA(model, beta=0.999)

    # Load the model from the checkpoint
    print(f"Loading model from {args.checkpoint}")
    ema_model.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    # Save the checkpoint
    print(f"Saving model to {args.output}")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hyperparams": params,
    }
    torch.save(checkpoint, args.output)