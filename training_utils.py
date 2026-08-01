from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_splitter import build_artifacts

KAGGLE_DATASET = "hasyimabdillah/workoutexercises-images"


def ensure_dataset(project_root: Path) -> list[str]:
    """Make sure project_root/data has class folders, downloading from Kaggle if needed."""
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not any(data_dir.glob("*/*")):
        import kagglehub

        access_token_path = Path.home() / ".kaggle" / "access_token"
        kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"

        if not (os.environ.get("KAGGLE_API_TOKEN") or access_token_path.exists() or kaggle_json_path.exists()):
            try:
                from google.colab import userdata

                os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
            except Exception:
                # userdata secrets need the actual Colab browser UI; when the kernel is
                # driven from VS Code (or there's no KAGGLE_API_TOKEN secret), fall back
                # to an interactive prompt. Nothing typed here is written to disk.
                os.environ["KAGGLE_API_TOKEN"] = getpass.getpass("Kaggle API token: ")

        print("Downloading dataset from Kaggle...")
        cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))

        # kagglehub caches the dataset under its own path; if it's wrapped in one extra
        # top-level folder, look inside that instead of the cache root.
        source = cache_path
        entries = list(source.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            source = entries[0]

        for class_dir in source.iterdir():
            if class_dir.is_dir():
                target = data_dir / class_dir.name
                if not target.exists():
                    target.symlink_to(class_dir, target_is_directory=True)

    return sorted(p.name for p in data_dir.iterdir() if p.is_dir())


def ensure_artifacts(cfg: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Build (or reuse) the split/manifest CSVs that dataloaders read from."""
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


def save_results(
    trainer: pl.Trainer,
    artifacts: dict[str, Path],
    test_results: list[dict[str, float]],
    prediction_batches: list[dict[str, Any]],
    project_root: Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten predictions to CSV and write a training_summary.json; returns the summary dict.

    Persists cfg['model'] (the architecture config, e.g. which backbone) alongside the
    checkpoint path, so 05_inference.ipynb can rebuild the exact matching architecture
    later even if configs/base.yaml's defaults have since changed.
    """
    prediction_frame = _flatten_prediction_batches(prediction_batches)
    prediction_path = artifacts["sequence_manifest"].parent / "predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    summary = {
        "project_root": str(project_root),
        "best_model_path": trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else "",
        "test_results": test_results,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "predictions_path": str(prediction_path),
        "model_config": (cfg or {}).get("model", {}),
    }

    summary_path = artifacts["sequence_manifest"].parent / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved training summary to {summary_path}")
    print(f"Saved predictions to {prediction_path}")
    return summary


def classification_metrics(results: pd.DataFrame, class_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Confusion matrix and per-class precision/recall/F1 from a predictions dataframe
    with integer 'label'/'prediction' columns (predictions.csv or evaluate_multi_clip output).
    class_names must be ordered by label_id (0..N-1)."""
    from sklearn.metrics import classification_report, confusion_matrix

    labels = list(range(len(class_names)))
    confusion_df = pd.DataFrame(
        confusion_matrix(results["label"], results["prediction"], labels=labels),
        index=class_names,
        columns=class_names,
    )

    report = classification_report(
        results["label"],
        results["prediction"],
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    return confusion_df, report_df


def plot_confusion_matrix(confusion_df: pd.DataFrame, title: str = "Confusion matrix"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(confusion_df.values, cmap="Blues")
    ax.set_xticks(range(len(confusion_df.columns)))
    ax.set_xticklabels(confusion_df.columns, rotation=90)
    ax.set_yticks(range(len(confusion_df.index)))
    ax.set_yticklabels(confusion_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def epoch_history(trainer: pl.Trainer) -> pd.DataFrame:
    """Per-epoch train/val loss+acc from the CSVLogger, for a clean summary table."""
    trainer.logger.save()  # force-flush any buffered rows before reading

    metrics_path = Path(trainer.logger.log_dir) / "metrics.csv"
    if not metrics_path.exists():
        # trainer.logger.log_dir's version can shift between when training started and
        # when this is called; fall back to the most recently modified metrics.csv
        # under the logger's root directory.
        candidates = sorted(
            Path(trainer.logger.root_dir).glob("*/metrics.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No metrics.csv found under {trainer.logger.root_dir}")
        metrics_path = candidates[-1]

    history = pd.read_csv(metrics_path)
    cols = [c for c in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"] if c in history.columns]
    return history[cols].groupby("epoch").last().reset_index()


@torch.no_grad()
def evaluate_multi_clip(
    lit_module,
    dataset,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[float, pd.DataFrame]:
    """Average softmax predictions over each clip's multiple windows (see MultiClipWorkoutDataset).

    Returns (accuracy, per-clip results) so it can be compared directly against the
    single-window accuracy from trainer.test()/epoch_history() on the same split.
    """
    device = next(lit_module.parameters()).device
    was_training = lit_module.training
    lit_module.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows: list[dict[str, Any]] = []

    for batch in loader:
        frames = batch["frames"].to(device)  # (B, num_clips, sequence_len, C, H, W)
        num_windows = frames.shape[1]
        logits = lit_module(frames.reshape(-1, *frames.shape[2:]))
        probs = F.softmax(logits, dim=1).reshape(frames.shape[0], num_windows, -1).mean(dim=1)
        confidence, prediction = probs.max(dim=1)

        for i in range(frames.shape[0]):
            label = int(batch["label"][i])
            pred = int(prediction[i])
            rows.append(
                {
                    "clip_id": batch["clip_id"][i],
                    "class_name": batch["class_name"][i],
                    "label": label,
                    "prediction": pred,
                    "confidence": float(confidence[i]),
                    "correct": label == pred,
                }
            )

    lit_module.train(was_training)
    results = pd.DataFrame(rows)
    accuracy = float(results["correct"].mean())
    return accuracy, results
