encoder_layers_matrix = [2, 5]
encoder_layers = 2  # TODO sample

context_length_matrix = [100, 200]
context_length = 100  # TODO sample

config_matrix = {
    "action_context_length": 100,
    "batch_size": 64,
    "distill_teacher_inference_steps": 30,
    "encoder_patch_size": [1, 5],
    "epochs": 750,
    "hidden_dim": [256, 512, 1024],
    "image_context_length": context_length // 10,
    "image_encoder_type": ["resnet18", "swin_transformer_small"],
    "image_resolution": 224,
    "image_sequence_encoder_type": ["transformer", "none"],
    "image_use_final_avgpool": False,
    "imu_context_length": context_length,
    "imu_orientation_embedding_method": ["quaternion", "five_dim"],
    "joint_state_context_length": context_length,
    "joint_state_encoder_layers": encoder_layers,
    "lr": 1.0e-4,
    "num_action_history_encoder_layers": encoder_layers,
    "num_decoder_layers": [4, 10],
    "num_image_sequence_encoder_layers": encoder_layers,
    "num_imu_encoder_layers": encoder_layers,
    "num_joints": 20,
    "num_normalization_samples": 1000,
    "train_denoising_timesteps": [500, 1000],
    "trajectory_prediction_length": [5, 10, 20],
    "use_action_history": True,
    "use_gamestate": True,
    "use_images": True,
    "use_imu": True,
    "use_joint_states": True,
}
