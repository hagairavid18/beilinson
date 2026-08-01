from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class WorkoutSequenceDataset(Dataset):
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

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def _resize(self, image: Image.Image) -> Image.Image:
        if isinstance(self.image_size, int):
            size = (self.image_size, self.image_size)
        else:
            size = tuple(self.image_size)
        return image.resize(size, Image.BILINEAR)

    def _load_image(self, relative_path: str) -> torch.Tensor:
        path = self.data_root / relative_path
        with Image.open(path) as image:
            image = self._resize(image.convert("RGB"))
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.normalize:
            tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        frames = [self._load_image(row[column]) for column in self.frame_cols]
        frames_tensor = torch.stack(frames, dim=0)
        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return {
            "frames": frames_tensor,
            "label": label,
            "clip_id": str(row["clip_id"]),
            "class_name": str(row["class"]),
            "split": str(row["split"]),
        }
