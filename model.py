from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class FrameEncoder(nn.Module):
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


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        hidden_dims: Iterable[int] = (32, 64, 128),
        embedding_dim: int = 128,
        dropout: float = 0.2,
        temporal_pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.temporal_pooling = temporal_pooling.lower()
        self.frame_encoder = FrameEncoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

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
