from __future__ import annotations

from typing import Iterable

import torch
import torchvision.models as tv_models
from torch import nn

# Recommended for small datasets like this one (~1100 clips, 22 classes): MobileNetV3-Small
# and EfficientNet-B0 are the strongest lightweight transfer-learning backbones in practice -
# fast, few params relative to accuracy, and less prone to overfitting a small dataset than
# larger nets. ResNet18 is included as a heavier, well-established alternative.
_PRETRAINED_BACKBONES = {"resnet18", "mobilenet_v3_small", "efficientnet_b0"}


def _build_pretrained_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Returns (feature extractor with its classification head removed, feature_dim).

    pretrained=False skips downloading ImageNet weights - use when the caller is about
    to load its own checkpoint over the whole model anyway (e.g. inference notebooks),
    since the downloaded weights would just get overwritten.
    """
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tv_models.resnet18(weights=weights)
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feature_dim
    if name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = tv_models.mobilenet_v3_small(weights=weights)
        feature_dim = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return net, feature_dim
    if name == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tv_models.efficientnet_b0(weights=weights)
        feature_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, feature_dim
    raise ValueError(f"Unknown pretrained backbone={name!r}, expected one of {sorted(_PRETRAINED_BACKBONES)}")


class FrameEncoder(nn.Module):
    """Small from-scratch CNN (the original backbone='custom' option)."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: Iterable[int] = (32, 64, 128),
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for hidden_dim in hidden_dims:
            blocks.extend(
                [
                    nn.Conv2d(current_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            current_channels = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return self.projection(x)


class PretrainedFrameEncoder(nn.Module):
    """ImageNet-pretrained torchvision backbone + a projection head to embedding_dim.

    Expects ImageNet-normalized input (dataset.py already normalizes with ImageNet
    mean/std regardless of backbone, so no dataset changes are needed to use this).
    """

    def __init__(
        self,
        backbone: str,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        freeze: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = _build_pretrained_backbone(backbone, pretrained=pretrained)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        hidden_dims: Iterable[int] = (32, 64, 128),
        embedding_dim: int = 128,
        dropout: float = 0.2,
        temporal_pooling: str = "mean",
        backbone: str = "custom",
        freeze_backbone: bool = True,
        pretrained_backbone: bool = True,
        classifier_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.temporal_pooling = temporal_pooling.lower()

        if backbone == "custom":
            self.frame_encoder = FrameEncoder(
                in_channels=in_channels,
                hidden_dims=hidden_dims,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        elif backbone in _PRETRAINED_BACKBONES:
            self.frame_encoder = PretrainedFrameEncoder(
                backbone=backbone,
                embedding_dim=embedding_dim,
                dropout=dropout,
                freeze=freeze_backbone,
                pretrained=pretrained_backbone,
            )
        else:
            raise ValueError(f"Unknown backbone={backbone!r}, expected 'custom' or one of {sorted(_PRETRAINED_BACKBONES)}")

        if self.temporal_pooling == "lstm":
            hidden_size = max(16, embedding_dim // 2)
            self.temporal = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            feature_dim = hidden_size * 2
        else:
            self.temporal = None
            feature_dim = embedding_dim

        if classifier_hidden_dim:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, classifier_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )

    def _aggregate_temporal(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.temporal_pooling == "mean":
            return embeddings.mean(dim=1)
        if self.temporal_pooling == "max":
            return embeddings.max(dim=1).values
        if self.temporal_pooling == "lstm":
            _, (hidden_state, _) = self.temporal(embeddings)
            forward_last = hidden_state[-2]
            backward_last = hidden_state[-1]
            return torch.cat([forward_last, backward_last], dim=1)
        raise ValueError(f"Unknown temporal_pooling={self.temporal_pooling!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = x.shape[:2]
        frames = x.reshape(batch_size * sequence_length, *x.shape[2:])
        frame_embeddings = self.frame_encoder(frames).reshape(batch_size, sequence_length, -1)
        sequence_embedding = self._aggregate_temporal(frame_embeddings)
        return self.classifier(sequence_embedding)
