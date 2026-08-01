from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "data").is_dir() and (path / "configs").is_dir():
            return path
    for path in [start, *start.parents]:
        if (path / "data").is_dir():
            return path
    return start


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_clip_frame_token(path: Path, class_name: str) -> tuple[str, float, str]:
    token = path.stem.rsplit("_", 1)[-1]
    if token.isdigit():
        number = int(token)
        clip_id = f"{class_name}|num_{number // 10000}"
        frame_idx = float(number % 10000)
        return clip_id, frame_idx, "numeric"

    match = re.fullmatch(r"([A-Za-z]+)(\d+)", token)
    if match:
        prefix, number = match.groups()
        clip_id = f"{class_name}|{prefix}{number}"
        return clip_id, float("nan"), "alpha_num"

    return f"{class_name}|{token}", float("nan"), "other"


def length_bucket(num_frames: int) -> str:
    if num_frames <= 2:
        return "1-2"
    if num_frames <= 5:
        return "3-5"
    if num_frames <= 10:
        return "6-10"
    if num_frames <= 20:
        return "11-20"
    if num_frames <= 50:
        return "21-50"
    return "51+"


def resolution_bucket(width: int, height: int) -> str:
    area = width * height
    if area <= 100_000:
        return "tiny"
    if area <= 250_000:
        return "small"
    if area <= 600_000:
        return "medium"
    return "large"


def sample_or_pad_indices(num_frames: int, sequence_len: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros(sequence_len, dtype=int)
    if num_frames >= sequence_len:
        return np.linspace(0, num_frames - 1, sequence_len, dtype=int)
    indices = np.arange(num_frames, dtype=int)
    pad = np.full(sequence_len - num_frames, num_frames - 1, dtype=int)
    return np.concatenate([indices, pad])


def _collect_frame_rows(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_dir in sorted([path for path in data_dir.iterdir() if path.is_dir()]):
        class_name = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            clip_id, frame_idx, token_type = parse_clip_frame_token(image_path, class_name)
            rows.append(
                {
                    "class": class_name,
                    "file": image_path.name,
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "token_type": token_type,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No image files found under {data_dir}")
    return pd.DataFrame(rows)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def build_clip_meta(df_frames: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (class_name, clip_id), group in df_frames.groupby(["class", "clip_id"]):
        group = group.sort_values(["frame_idx", "file"], na_position="last")
        first_file = group.iloc[0]["file"]
        image_path = data_dir / class_name / first_file
        width, height = _image_size(image_path)
        num_frames = len(group)
        rows.append(
            {
                "class": class_name,
                "clip_id": clip_id,
                "n_frames": num_frames,
                "width": width,
                "height": height,
                "length_bucket": length_bucket(num_frames),
                "resolution_bucket": resolution_bucket(width, height),
                "bucket_key": f"{length_bucket(num_frames)}|{resolution_bucket(width, height)}",
            }
        )
    return pd.DataFrame(rows)


def assign_split_frames(
    clip_meta: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []

    for class_name, group in clip_meta.groupby("class"):
        bucket_lists: list[list[str]] = []
        for _, bucket_group in group.groupby("bucket_key"):
            clip_ids = bucket_group["clip_id"].to_numpy().copy()
            rng.shuffle(clip_ids)
            bucket_lists.append(list(clip_ids))

        ordered_clip_ids: list[str] = []
        while any(bucket_lists):
            for bucket_ids in bucket_lists:
                if bucket_ids:
                    ordered_clip_ids.append(bucket_ids.pop())

        total = len(ordered_clip_ids)
        n_train = int(round(train_frac * total))
        n_val = int(round(val_frac * total))
        if total >= 3:
            n_train = max(1, min(n_train, total - 2))
            n_val = max(1, min(n_val, total - n_train - 1))
        split = np.array(["test"] * total, dtype=object)
        split[:n_train] = "train"
        split[n_train : n_train + n_val] = "val"
        parts.append(pd.DataFrame({"class": class_name, "clip_id": ordered_clip_ids, "split": split}))

    split_clips = pd.concat(parts, ignore_index=True)
    return split_clips


def build_sequence_manifest(df_frames_split: pd.DataFrame, sequence_len: int, data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_cols = [f"f{i:02d}" for i in range(sequence_len)]

    for (class_name, clip_id, split), group in df_frames_split.groupby(["class", "clip_id", "split"]):
        ordered = group.sort_values(["frame_idx", "file"], na_position="last")
        rel_paths = [f"data/{class_name}/{filename}" for filename in ordered["file"].tolist()]
        chosen = sample_or_pad_indices(len(rel_paths), sequence_len)
        first_row = ordered.iloc[0]
        row = {
            "class": class_name,
            "clip_id": clip_id,
            "split": split,
            "n_frames_orig": len(rel_paths),
            "frame_idx_min": ordered["frame_idx"].min(skipna=True),
            "frame_idx_max": ordered["frame_idx"].max(skipna=True),
            "token_type": first_row.get("token_type", "unknown"),
        }
        for idx, frame_col in enumerate(frame_cols):
            row[frame_col] = rel_paths[chosen[idx]]
        rows.append(row)

    sequence_manifest = pd.DataFrame(rows)
    label_map = pd.DataFrame({"class": sorted(sequence_manifest["class"].unique())})
    label_map["label_id"] = np.arange(len(label_map), dtype=int)

    sequence_manifest = sequence_manifest.merge(label_map, on="class", how="left")
    return sequence_manifest, label_map


def build_artifacts(
    data_dir: str | Path,
    artifacts_dir: str | Path,
    sequence_len: int,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> dict[str, Path]:
    data_dir = Path(data_dir).resolve()
    artifacts_dir = Path(artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df_frames = _collect_frame_rows(data_dir)
    clip_meta = build_clip_meta(df_frames, data_dir)
    split_clips = assign_split_frames(clip_meta, seed, train_frac, val_frac, test_frac)
    df_frames_split = df_frames.merge(split_clips, on=["class", "clip_id"], how="left")
    clip_meta_split = clip_meta.merge(split_clips, on=["class", "clip_id"], how="left")
    sequence_manifest, label_map = build_sequence_manifest(df_frames_split, sequence_len, data_dir)

    if sequence_manifest["label_id"].isna().any():
        raise RuntimeError("Label mapping failed while building the sequence manifest")

    split_clips_path = artifacts_dir / "clip_splits.csv"
    frame_manifest_path = artifacts_dir / "frame_manifest_with_split.csv"
    clip_meta_path = artifacts_dir / "clip_meta_with_split.csv"
    sequence_manifest_path = artifacts_dir / f"sequence_manifest_len{sequence_len}.csv"
    label_map_path = artifacts_dir / "label_map.csv"

    split_clips.to_csv(split_clips_path, index=False)
    df_frames_split.to_csv(frame_manifest_path, index=False)
    clip_meta_split.to_csv(clip_meta_path, index=False)
    sequence_manifest.to_csv(sequence_manifest_path, index=False)
    label_map.to_csv(label_map_path, index=False)

    return {
        "split_clips": split_clips_path,
        "frame_manifest": frame_manifest_path,
        "clip_meta": clip_meta_path,
        "sequence_manifest": sequence_manifest_path,
        "label_map": label_map_path,
    }


def _config_args(config: dict[str, Any]) -> dict[str, Any]:
    mode = config.get("mode", {})
    data = config.get("data", {})
    return {
        "seed": int(mode.get("seed", 42)),
        "train_frac": float(data.get("train_frac", 0.70)),
        "val_frac": float(data.get("val_frac", 0.15)),
        "test_frac": float(data.get("test_frac", 0.15)),
        "sequence_len": int(data.get("sequence_len", 16)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test manifests from the extracted workout frames.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--sequence-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--val-frac", type=float, default=None)
    parser.add_argument("--test-frac", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = find_project_root(Path(args.config).resolve().parent)
    config_args = _config_args(config)

    data_cfg = config.get("data", {})
    data_dir = args.data_dir or data_cfg.get("data_dir", "data")
    artifacts_dir = args.artifacts_dir or data_cfg.get("artifacts_dir", "artifacts")
    sequence_len = args.sequence_len or config_args["sequence_len"]
    seed = args.seed if args.seed is not None else config_args["seed"]
    train_frac = args.train_frac if args.train_frac is not None else config_args["train_frac"]
    val_frac = args.val_frac if args.val_frac is not None else config_args["val_frac"]
    test_frac = args.test_frac if args.test_frac is not None else config_args["test_frac"]

    data_path = resolve_path(project_root, data_dir)
    artifacts_path = resolve_path(project_root, artifacts_dir)

    outputs = build_artifacts(
        data_dir=data_path,
        artifacts_dir=artifacts_path,
        sequence_len=sequence_len,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    for name, path in outputs.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
