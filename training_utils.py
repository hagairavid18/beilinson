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
from PIL import Image
from torch.utils.data import DataLoader

from data_splitter import IMAGE_EXTS, build_artifacts

KAGGLE_DATASET = "hasyimabdillah/workoutexercises-images"


def launch_tensorboard(logdir: Path, port: int = 6006) -> str:
    """Start (or reuse) a TensorBoard server for logdir and open it in its own browser
    tab/window instead of embedding it inline in the notebook output, which gets slow
    and cramped over a long training run. On Colab this opens via the proxied port
    (a real new tab); locally it opens the default browser at localhost.
    """
    import webbrowser

    from tensorboard import program

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(logdir), "--port", str(port)])
    url = tb.launch()

    try:
        from google.colab import output

        output.serve_kernel_port_as_window(port)
    except ImportError:
        webbrowser.open(url)

    return url


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


def ensure_image_cache(project_root: Path, image_size: int) -> Path:
    """Pre-resize every image once into artifacts/image_cache_<size>/data/<class>/<file>,
    mirroring data/'s layout so the returned path is a drop-in data_root for
    WorkoutSequenceDataset/MultiClipWorkoutDataset. Skips files already cached, so it's
    cheap on repeat runs. Benchmarked ~1.6x faster per-image load than decoding the
    (often much larger) original every epoch.
    """
    data_dir = project_root / "data"
    cache_root = project_root / "artifacts" / f"image_cache_{image_size}"
    cache_data_dir = cache_root / "data"

    image_paths = [p for p in data_dir.glob("*/*") if p.suffix.lower() in IMAGE_EXTS]
    to_cache = [p for p in image_paths if not (cache_data_dir / p.relative_to(data_dir)).exists()]

    if to_cache:
        print(f"Caching {len(to_cache)} images at {image_size}x{image_size} (one-time)...")
        for path in to_cache:
            target = cache_data_dir / path.relative_to(data_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                image.convert("RGB").resize((image_size, image_size), Image.BILINEAR).save(target, quality=90)

    return cache_root


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
    output_dir: Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten predictions to CSV and write a training_summary.json under output_dir
    (typically the current experiment's artifacts/<exp_name>/ folder); returns the summary
    dict. Persists cfg['model'] (the architecture config, e.g. which backbone) alongside the
    checkpoint path, so 05_inference.ipynb can rebuild the exact matching architecture later
    even if configs/base.yaml's defaults have since changed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame = _flatten_prediction_batches(prediction_batches)
    prediction_path = output_dir / "predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    summary = {
        "project_root": str(project_root),
        "best_model_path": trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else "",
        "test_results": test_results,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "predictions_path": str(prediction_path),
        "model_config": (cfg or {}).get("model", {}),
    }

    summary_path = output_dir / "training_summary.json"
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


def _run_training(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Runs one already-loaded config's full pipeline (dataloaders -> model ->
    Trainer.fit) end to end and returns {exp_name, fold_index, best_val_acc,
    best_checkpoint}. Mirrors 03_train_colab.ipynb's individual cells. Takes a dict
    (not a path) so run_kfold_sweep can inject a different fold_index/exp_name per
    fold without writing a separate yaml file per fold.
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    from dataset import WorkoutSequenceDataset
    from model import SequenceClassifier
    from pytorch_lightning import WorkoutLightningModule

    exp_name = cfg["exp_name"]
    exp_dir = project_root / "artifacts" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(cfg["mode"]["seed"], workers=True)
    artifacts = ensure_artifacts(cfg, project_root)

    data_cfg = cfg["data"]
    cached_data_root = ensure_image_cache(project_root, data_cfg["image_size"])

    train_dataset = WorkoutSequenceDataset(
        artifacts["sequence_manifest"], cached_data_root, split="train", image_size=data_cfg["image_size"],
        augment=data_cfg.get("augment", False),
        frame_manifest_path=artifacts["frame_manifest"],
        label_map_path=artifacts["label_map"],
        sequence_len=data_cfg["sequence_len"],
    )
    val_dataset = WorkoutSequenceDataset(
        artifacts["sequence_manifest"], cached_data_root, split="val", image_size=data_cfg["image_size"]
    )

    num_workers = data_cfg["num_workers"]
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=bool(num_workers),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=data_cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=bool(num_workers),
    )

    label_map = pd.read_csv(artifacts["label_map"])
    num_classes = int(label_map["label_id"].nunique())

    model_cfg, training_cfg = cfg["model"], cfg["training"]
    model = SequenceClassifier(
        num_classes=num_classes,
        in_channels=model_cfg["in_channels"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg["dropout"],
        temporal_pooling=model_cfg["temporal_pooling"],
        backbone=model_cfg.get("backbone", "custom"),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        classifier_hidden_dim=model_cfg.get("classifier_hidden_dim"),
    )
    lit_module = WorkoutLightningModule(
        model=model,
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
        lr_step_size=training_cfg.get("lr_step_size"),
        lr_gamma=training_cfg.get("lr_gamma", 0.5),
    )

    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = training_cfg.get("monitor", "val_acc")
    monitor_mode = training_cfg.get("monitor_mode", "max")

    precision = training_cfg.get("precision", "32-true")
    if precision == "auto":
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = pl.Trainer(
        max_epochs=training_cfg.get("max_epochs", 10),
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", "auto"),
        precision=precision,
        log_every_n_steps=training_cfg.get("log_every_n_steps", 10),
        default_root_dir=str(exp_dir),
        logger=[CSVLogger(save_dir=str(exp_dir)), TensorBoardLogger(save_dir=str(exp_dir), name="tb_logs")],
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir, filename="epoch{epoch:02d}-{val_acc:.3f}",
                monitor=monitor, mode=monitor_mode, save_top_k=1,
            ),
            EarlyStopping(monitor=monitor, mode=monitor_mode, patience=training_cfg.get("patience", 4)),
        ],
    )
    trainer.fit(lit_module, train_loader, val_loader)

    return {
        "exp_name": exp_name,
        "fold_index": data_cfg.get("fold_index"),
        "best_val_acc": float(trainer.checkpoint_callback.best_model_score),
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
    }


def run_config(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Loads a yaml config and runs its full pipeline end to end (see _run_training)."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return _run_training(cfg, project_root)


def run_kfold_sweep(config_path: Path, project_root: Path) -> pd.DataFrame:
    """Runs one config's `data.n_folds` folds back-to-back, injecting a different
    fold_index (and an exp_name suffixed with it) into an in-memory copy of the config
    each time - no need for a separate yaml file per fold. Each fold still gets its own
    artifacts/<exp_name>_fold<i>/ folder, like any other run. Returns one row per fold
    (exp_name, fold_index, best_val_acc, best_checkpoint).
    """
    import copy

    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    n_folds = base_cfg["data"].get("n_folds")
    if not n_folds:
        raise ValueError(f"{config_path} has no data.n_folds set - nothing to sweep over")

    results = []
    for fold_index in range(n_folds):
        fold_cfg = copy.deepcopy(base_cfg)
        fold_cfg["data"]["fold_index"] = fold_index
        fold_cfg["exp_name"] = f"{base_cfg['exp_name']}_fold{fold_index}"
        print(f"--- {fold_cfg['exp_name']} (fold {fold_index + 1}/{n_folds}) ---")
        results.append(_run_training(fold_cfg, project_root))

    results_df = pd.DataFrame(results)
    mean_acc, std_acc = results_df["best_val_acc"].mean(), results_df["best_val_acc"].std()
    print(f"\nMean val_acc across {n_folds} folds: {mean_acc:.4f} ± {std_acc:.4f}")
    return results_df
