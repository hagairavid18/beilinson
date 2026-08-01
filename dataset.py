from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _resize(image: Image.Image, image_size: int | tuple[int, int]) -> Image.Image:
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    return image.resize(size, Image.BILINEAR)


def load_image(
    path: str | Path,
    image_size: int | tuple[int, int],
    normalize: bool = True,
) -> torch.Tensor:
    with Image.open(path) as image:
        image = _resize(image.convert("RGB"), image_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def _load_clips(frame_manifest_path: str | Path, label_map_path: str | Path, split: str) -> list[dict[str, Any]]:
    """Every clip in `split`, as {class, clip_id, label_id, files} - files is the clip's
    full ordered list of original frame paths (not the sequence manifest's fixed subset)."""
    frames = pd.read_csv(frame_manifest_path)
    frames = frames[frames["split"] == split]
    if frames.empty:
        raise ValueError(f"No rows found in {frame_manifest_path} for split={split!r}")

    label_map = pd.read_csv(label_map_path)
    frames = frames.merge(label_map, on="class", how="left")

    clips: list[dict[str, Any]] = []
    for (class_name, clip_id), group in frames.groupby(["class", "clip_id"]):
        ordered = group.sort_values(["frame_idx", "file"], na_position="last")
        clips.append(
            {
                "class": class_name,
                "clip_id": clip_id,
                "label_id": int(ordered["label_id"].iloc[0]),
                "files": [f"data/{class_name}/{filename}" for filename in ordered["file"].tolist()],
            }
        )
    return clips


class WorkoutSequenceDataset(Dataset):
    """One fixed-length (sequence_len-frame) sample per clip.

    By default, reads the sequence manifest's fixed frame columns (f00..f15), chosen once
    when the manifest was built - deterministic, used for val/test.

    With augment=True (train split only), instead re-samples sequence_len frames from the
    clip's full frame set on every __getitem__ call - temporal jitter via
    sample_random_indices - and randomly mirrors the whole clip (consistently across all
    its frames, to keep the sequence temporally coherent). Needs frame_manifest_path and
    label_map_path (from ensure_artifacts) instead of the fixed sequence manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str | None = None,
        image_size: int | tuple[int, int] = 128,
        normalize: bool = True,
        augment: bool = False,
        frame_manifest_path: str | Path | None = None,
        label_map_path: str | Path | None = None,
        sequence_len: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment

        if augment:
            if frame_manifest_path is None or label_map_path is None or sequence_len is None:
                raise ValueError("augment=True requires frame_manifest_path, label_map_path, and sequence_len")
            self.sequence_len = sequence_len
            self.clips = _load_clips(frame_manifest_path, label_map_path, split)
            return

        self.manifest_path = Path(manifest_path)
        df = pd.read_csv(self.manifest_path)
        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows found in {self.manifest_path} for split={split!r}")

        self.df = df
        self.frame_cols = sorted([column for column in self.df.columns if re.fullmatch(r"f\d\d", column)])
        if not self.frame_cols:
            raise ValueError(f"Manifest {self.manifest_path} does not contain fixed-length frame columns")

    def __len__(self) -> int:
        return len(self.clips) if self.augment else len(self.df)

    def _getitem_augmented(self, index: int) -> dict[str, Any]:
        from data_splitter import sample_random_indices

        clip = self.clips[index]
        rng = np.random.default_rng()
        indices = sample_random_indices(len(clip["files"]), self.sequence_len, rng)
        frames = [load_image(self.data_root / clip["files"][i], self.image_size, self.normalize) for i in indices]
        frames_tensor = torch.stack(frames, dim=0)
        if rng.random() < 0.5:
            frames_tensor = torch.flip(frames_tensor, dims=[-1])
        return {
            "frames": frames_tensor,
            "label": torch.tensor(clip["label_id"], dtype=torch.long),
            "clip_id": str(clip["clip_id"]),
            "class_name": str(clip["class"]),
            "split": "train",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.augment:
            return self._getitem_augmented(index)

        row = self.df.iloc[index]
        frames = [
            load_image(self.data_root / row[column], self.image_size, self.normalize) for column in self.frame_cols
        ]
        frames_tensor = torch.stack(frames, dim=0)
        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return {
            "frames": frames_tensor,
            "label": label,
            "clip_id": str(row["clip_id"]),
            "class_name": str(row["class"]),
            "split": str(row["split"]),
        }


class MultiClipWorkoutDataset(Dataset):
    """num_clips windows of sequence_len frames per clip, evenly covering the full clip.

    Built directly from the per-frame manifest (frame_manifest_with_split.csv) rather than
    the fixed sequence manifest, so it has access to every original frame per clip - used to
    check whether averaging predictions over multiple windows beats the single-window sample
    each clip gets during training/standard evaluation.
    """

    def __init__(
        self,
        frame_manifest_path: str | Path,
        label_map_path: str | Path,
        data_root: str | Path,
        split: str,
        sequence_len: int = 16,
        num_clips: int = 5,
        image_size: int | tuple[int, int] = 128,
        normalize: bool = True,
    ) -> None:
        from data_splitter import sample_or_pad_indices_multi

        self._sample_or_pad_indices_multi = sample_or_pad_indices_multi
        self.data_root = Path(data_root)
        self.sequence_len = sequence_len
        self.num_clips = num_clips
        self.image_size = image_size
        self.normalize = normalize
        self.clips = _load_clips(frame_manifest_path, label_map_path, split)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clips[index]
        files = clip["files"]
        windows = self._sample_or_pad_indices_multi(len(files), self.sequence_len, self.num_clips)

        clip_frames = []
        for window in windows:
            frames = [load_image(self.data_root / files[i], self.image_size, self.normalize) for i in window]
            clip_frames.append(torch.stack(frames, dim=0))

        return {
            "frames": torch.stack(clip_frames, dim=0),  # (num_clips, sequence_len, C, H, W)
            "label": torch.tensor(clip["label_id"], dtype=torch.long),
            "clip_id": str(clip["clip_id"]),
            "class_name": str(clip["class"]),
        }
