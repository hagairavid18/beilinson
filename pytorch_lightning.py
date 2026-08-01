from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import lightning.pytorch as pl


class WorkoutLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_step_size: int | None = None,
        lr_gamma: float = 0.5,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.label_smoothing = label_smoothing
        # Not a buffer/param (it's derived data, not learned) - moved to the right device
        # manually in _step instead, so it round-trips through load_from_checkpoint fine
        # (class_weights isn't saved; predictions/accuracy don't depend on it anyway).
        self.class_weights = class_weights

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model(frames)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        frames = batch["frames"]
        labels = batch["label"]
        logits = self(frames)
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(logits, labels, weight=weight, label_smoothing=self.label_smoothing)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        frames = batch["frames"]
        labels = batch["label"]
        logits = self(frames)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        return {
            "clip_id": batch["clip_id"],
            "class_name": batch["class_name"],
            "label": labels.detach().cpu(),
            "prediction": predictions.detach().cpu(),
            "confidence": confidence.detach().cpu(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if not self.lr_step_size:
            return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
