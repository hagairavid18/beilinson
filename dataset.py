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


class WorkoutSequenceDataset(Dataset):
    """One fixed-length (sequence_len-frame) sample per clip, from the sequence manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str | None = None,
        image_size: int | tuple[int, int] = 128,
        normalize: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.normalize = normalize

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
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
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

        frames = pd.read_csv(frame_manifest_path)
        frames = frames[frames["split"] == split]
        if frames.empty:
            raise ValueError(f"No rows found in {frame_manifest_path} for split={split!r}")

        label_map = pd.read_csv(label_map_path)
        frames = frames.merge(label_map, on="class", how="left")

        self.clips: list[dict[str, Any]] = []
        for (class_name, clip_id), group in frames.groupby(["class", "clip_id"]):
            ordered = group.sort_values(["frame_idx", "file"], na_position="last")
            self.clips.append(
                {
                    "class": class_name,
                    "clip_id": clip_id,
                    "label_id": int(ordered["label_id"].iloc[0]),
                    "files": [f"data/{class_name}/{filename}" for filename in ordered["file"].tolist()],
                }
            )

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
