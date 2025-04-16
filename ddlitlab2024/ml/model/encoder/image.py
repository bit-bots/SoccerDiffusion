from enum import Enum

import torch
from torch import nn
from torchvision.models import resnet18, resnet50, swin_s, swin_t
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights

from ddlitlab2024.ml.model.encoder.base import BaseEncoder


class ImageEncoderType(Enum):
    """
    Enum class for the image encoder types.
    """

    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    SWIN_TRANSFORMER_TINY = "swin_transformer_tiny"
    SWIN_TRANSFORMER_SMALL = "swin_transformer_small"
    DINOV2 = "dinov2"


class SequenceEncoderType(Enum):
    """
    Enum class for the sequence encoder types.
    """

    TRANSFORMER = "transformer"
    NONE = "none"


class AbstractImageEncoder(nn.Module):
    """
    Abstract class for image encoders.
    """

    encoder: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the image encoder.

        :param x: A sequence of images.
        :return: A sequence of encoded images.
        """
        # Squash the sequence dimension together with the batch dimension
        images = x.view(-1, *x.shape[2:])
        
        # Encode the images into tokens
        tokens = self.encoder(images)

        # Restore the original sequence dimension
        return tokens.view(x.shape[0], x.shape[1], -1)


class ResNetImageEncoder(AbstractImageEncoder):
    """
    ResNet image encoder.
    """

    def __init__(self, resnet_type: ImageEncoderType, hidden_dim: int, use_final_avgpool: bool, resolution: int):
        super().__init__()
        match resnet_type:
            case ImageEncoderType.RESNET18:
                self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
            case ImageEncoderType.RESNET50:
                self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            case _:
                raise ValueError(f"Invalid ResNet type: {resnet_type}")
        if use_final_avgpool:
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, hidden_dim)
        else:
            self.encoder.avgpool = nn.Conv2d(self.encoder.fc.in_features, 32, 1)
            self.encoder.fc = nn.Linear(ResNetImageEncoder.calculate_output_size(resolution) ** 2 * 32, hidden_dim)

    @staticmethod
    def calculate_output_size(resolution):
        # Initial convolutional layer
        resolution = (resolution - 7 + 2 * 3) // 2 + 1
        # Max pooling layer
        resolution = (resolution - 3 + 2 * 1) // 2 + 1
        # Residual blocks (downsampling by a factor of 2 in each stage)
        resolution = resolution // 2 // 2 // 2
        return resolution


class SwinTransformerImageEncoder(AbstractImageEncoder):
    """
    Swin Transformer image encoder.
    """

    def __init__(self, swin_type: ImageEncoderType, hidden_dim: int):
        super().__init__()
        match swin_type:
            case ImageEncoderType.SWIN_TRANSFORMER_TINY:
                self.encoder = swin_t()
            case ImageEncoderType.SWIN_TRANSFORMER_SMALL:
                self.encoder = swin_s()
            case _:
                raise ValueError(f"Invalid Swin Transformer type: {swin_type}")
        self.encoder.head = nn.Linear(self.encoder.head.in_features, hidden_dim)


class DinoImageEncoder(AbstractImageEncoder):
    """
    Dino image encoder (dummy because we use precomputed embeddings for this)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(384, hidden_dim)


class TransformerImageSequenceEncoder(nn.Module):
    """
    Transformer image sequence encoder.
    """

    def __init__(self, image_encoder: AbstractImageEncoder, hidden_dim: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.image_encoder = image_encoder
        self.transformer_encoder = BaseEncoder(
            input_dim=hidden_dim,
            patch_size=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=8,
            max_seq_len=max_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(self.image_encoder(x))


def image_encoder_factory(
    encoder_type: ImageEncoderType, hidden_dim: int, use_final_avgpool: bool, resolution: int
) -> AbstractImageEncoder:
    """
    Factory function for creating image encoders.

    :param encoder_type: The type of the image encoder.
    :param hidden_dim: The number of hidden dimensions.
    :param use_final_avgpool: Whether to use the final average pooling layer.
    :param resolution: The resolution of the images.
    :return: The image encoder.
    """
    if encoder_type in [ImageEncoderType.RESNET18, ImageEncoderType.RESNET50]:
        return ResNetImageEncoder(encoder_type, hidden_dim, use_final_avgpool, resolution)
    if encoder_type in [ImageEncoderType.SWIN_TRANSFORMER_TINY, ImageEncoderType.SWIN_TRANSFORMER_SMALL]:
        return SwinTransformerImageEncoder(encoder_type, hidden_dim)
    if encoder_type in [ImageEncoderType.DINOV2]:
        return DinoImageEncoder(hidden_dim)
    else:
        raise ValueError(f"Invalid image encoder type: {encoder_type}")


def image_sequence_encoder_factory(
    encoder_type: SequenceEncoderType,
    image_encoder_type: ImageEncoderType,
    hidden_dim: int,
    num_layers: int,
    max_seq_len: int,
    use_final_avgpool: bool,
    resolution: int,
):
    """
    Factory function for creating image sequence encoders.

    :param encoder_type: The type of the sequence encoder that allows communication between different images.
        If no sequence encoder is needed, the image encoder is returned.
    :param image_encoder_type: The type of the image encoder.
    :param hidden_dim: The number of hidden dimensions.
    :param num_layers: The number of transformer layers.
    :param max_seq_len: The maximum sequence length.
    :param use_final_avgpool: Whether to use the final average pooling layer.
    :param resolution: The resolution of the images.
    :return: The image sequence encoder.
    """
    image_encoder = image_encoder_factory(image_encoder_type, hidden_dim, use_final_avgpool, resolution)

    match encoder_type:
        case SequenceEncoderType.TRANSFORMER:
            return TransformerImageSequenceEncoder(image_encoder, hidden_dim, num_layers, max_seq_len)
        case SequenceEncoderType.NONE:
            return image_encoder
        case _:
            raise ValueError(f"Invalid sequence encoder type: {encoder_type}")
