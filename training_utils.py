from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from data_splitter import build_artifacts
from dataset import WorkoutSequenceDataset
from model import SequenceClassifier
from pytorch_lightning import WorkoutLightningModule


def ensure_artifacts(cfg: dict[str, Any], project_root: Path) -> dict[str, Path]:
    mode_cfg = cfg.get("mode", {})
    data_cfg = cfg.get("data", {})
    data_dir = project_root / "data"
    artifacts_dir = project_root / "artifacts"
    sequence_len = int(data_cfg.get("sequence_len", 16))
    sequence_manifest = artifacts_dir / f"sequence_manifest_len{sequence_len}.csv"

    if bool(mode_cfg.get("build_artifacts", True)) or not sequence_manifest.exists():
        return build_artifacts(
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            sequence_len=sequence_len,
            seed=int(mode_cfg.get("seed", 42)),
            train_frac=float(data_cfg.get("train_frac", 0.70)),
            val_frac=float(data_cfg.get("val_frac", 0.15)),
            test_frac=float(data_cfg.get("test_frac", 0.15)),
        )

    return {
        "split_clips": artifacts_dir / "clip_splits.csv",
        "frame_manifest": artifacts_dir / "frame_manifest_with_split.csv",
        "clip_meta": artifacts_dir / "clip_meta_with_split.csv",
        "sequence_manifest": sequence_manifest,
        "label_map": artifacts_dir / "label_map.csv",
    }


def build_dataloaders(cfg: dict[str, Any], project_root: Path):
    data_cfg = cfg.get("data", {})
    artifacts = ensure_artifacts(cfg, project_root)

    # Manifest frame paths are already prefixed "data/<class>/<file>", relative
    # to project_root - so data_root here must be project_root, not project_root/'data'.
    data_root = project_root
    manifest_path = artifacts["sequence_manifest"]
    image_size = int(data_cfg.get("image_size", 128))
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = torch.cuda.is_available()

    train_dataset = WorkoutSequenceDataset(manifest_path, data_root, split="train", image_size=image_size)
    val_dataset = WorkoutSequenceDataset(manifest_path, data_root, split="val", image_size=image_size)
    test_dataset = WorkoutSequenceDataset(manifest_path, data_root, split="test", image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers),
    )

    label_map = pd.read_csv(artifacts["label_map"])
    num_classes = int(label_map["label_id"].nunique())
    return train_loader, val_loader, test_loader, num_classes, artifacts


def build_model(cfg: dict[str, Any], num_classes: int) -> SequenceClassifier:
    model_cfg = cfg.get("model", {})
    return SequenceClassifier(
        num_classes=num_classes,
        in_channels=int(model_cfg.get("in_channels", 3)),
        hidden_dims=tuple(model_cfg.get("hidden_dims", [32, 64, 128])),
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.20)),
        temporal_pooling=str(model_cfg.get("temporal_pooling", "mean")),
    )


def build_lightning_module(cfg: dict[str, Any], num_classes: int) -> WorkoutLightningModule:
    training_cfg = cfg.get("training", {})
    model = build_model(cfg, num_classes=num_classes)
    return WorkoutLightningModule(
        model=model,
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )


def build_trainer(cfg: dict[str, Any], project_root: Path) -> pl.Trainer:
    training_cfg = cfg.get("training", {})
    checkpoint_dir = project_root / training_cfg.get("checkpoint_dir", "artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor = str(training_cfg.get("monitor", "val_acc"))
    mode = str(training_cfg.get("monitor_mode", "max"))
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:02d}-{val_acc:.3f}",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=int(training_cfg.get("patience", 4)),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    precision = training_cfg.get("precision", "32-true")
    if precision == "auto":
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    return pl.Trainer(
        max_epochs=int(training_cfg.get("max_epochs", 10)),
        accelerator=str(training_cfg.get("accelerator", "auto")),
        devices=training_cfg.get("devices", "auto"),
        precision=precision,
        log_every_n_steps=int(training_cfg.get("log_every_n_steps", 10)),
        default_root_dir=str(project_root / "artifacts"),
        callbacks=callbacks,
    )


def _flatten_prediction_batches(prediction_batches: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch in prediction_batches:
        clip_ids = list(batch["clip_id"])
        class_names = list(batch["class_name"])
        labels = batch["label"].tolist()
        predictions = batch["prediction"].tolist()
        confidences = batch["confidence"].tolist()
        for clip_id, class_name, label, prediction, confidence in zip(
            clip_ids, class_names, labels, predictions, confidences
        ):
            rows.append(
                {
                    "clip_id": clip_id,
                    "class_name": class_name,
                    "label": int(label),
                    "prediction": int(prediction),
                    "confidence": float(confidence),
                    "correct": int(label) == int(prediction),
                }
            )
    return pd.DataFrame(rows)


def run_training(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    seed = int(cfg.get("mode", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    train_loader, val_loader, test_loader, num_classes, artifacts = build_dataloaders(cfg, project_root)
    lit_module = build_lightning_module(cfg, num_classes=num_classes)
    trainer = build_trainer(cfg, project_root)

    trainer.fit(lit_module, train_loader, val_loader)
    test_results = trainer.test(lit_module, dataloaders=test_loader, verbose=False)
    prediction_batches = trainer.predict(lit_module, dataloaders=test_loader)

    prediction_frame = _flatten_prediction_batches(prediction_batches)
    prediction_path = artifacts["sequence_manifest"].parent / "predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    metrics = {
        "project_root": str(project_root),
        "best_model_path": trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else "",
        "test_results": test_results,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "predictions_path": str(prediction_path),
    }

    metrics_path = artifacts["sequence_manifest"].parent / "training_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved training summary to {metrics_path}")
    print(f"Saved predictions to {prediction_path}")
    return metrics
