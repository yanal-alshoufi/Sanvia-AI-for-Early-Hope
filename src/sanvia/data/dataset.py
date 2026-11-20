"""PyTorch dataset for breast-level CC/MLO pairs and paired conservative augmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from sanvia.data.preprocessing import PNGPreprocessor


class PairedAugmenter:
    """Conservative mammography augmentation applied to the two views of one breast.

    Geometric and photometric parameters are sampled independently per view by default
    (CC and MLO are different projections); set ``share_params_across_views`` to share them.
    Photometric changes act only on non-zero pixels so the black background stays black.
    Labels are never modified. No flips, 90/180 degree rotations or cropping.
    """

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = dict(cfg)

    def _sample(self) -> dict[str, Any]:
        c = self.cfg
        u = random.uniform
        return {
            "geo": random.random() < c["p_geometric"],
            "angle": u(-c["rotation_degrees"], c["rotation_degrees"]),
            "tx": u(-c["translate_fraction"], c["translate_fraction"]),
            "ty": u(-c["translate_fraction"], c["translate_fraction"]),
            "scale": u(*c["scale_range"]),
            "photo": random.random() < c["p_photometric"],
            "brightness": u(*c["brightness_range"]),
            "contrast": u(*c["contrast_range"]),
            "gamma": u(*c["gamma_range"]),
            "noise": random.random() < c["p_noise"],
        }

    def _apply(self, x: torch.Tensor, p: Mapping[str, Any]) -> torch.Tensor:
        size = x.shape[-1]
        if p["geo"]:
            x = TF.affine(
                x,
                angle=p["angle"],
                translate=[round(p["tx"] * size), round(p["ty"] * size)],
                scale=p["scale"],
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=[0.0],
            )
        if p["photo"]:
            mask = x > 0
            if mask.any():
                mean = x[mask].mean()
                y = (x - mean) * p["contrast"] + mean
                y = y * p["brightness"]
                y = y.clamp(0.0, 1.0) ** p["gamma"]
                x = torch.where(mask, y, x)
        if p["noise"] and self.cfg["gaussian_noise_std"] > 0:
            mask = x > 0
            x = torch.where(mask, (x + torch.randn_like(x) * self.cfg["gaussian_noise_std"]).clamp(0.0, 1.0), x)
        return x

    def __call__(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.cfg.get("share_params_across_views", False):
            params = self._sample()
            return [self._apply(v, params) for v in views]
        return [self._apply(v, self._sample()) for v in views]


class BreastPairDataset(Dataset):
    """One item per breast: images ``[2, 3, S, S]`` (index 0 = CC, 1 = MLO), targets and explicit masks."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        preprocessor: PNGPreprocessor,
        augmenter: PairedAugmenter | None = None,
        age_features: np.ndarray | None = None,
    ):
        self.m = manifest.reset_index(drop=True)
        self.pre = preprocessor
        self.aug = augmenter
        self.age = None if age_features is None else torch.from_numpy(np.asarray(age_features, dtype=np.float32))
        if self.age is not None and len(self.age) != len(self.m):
            raise ValueError("age_features must have one row per manifest row.")
        self.birads = self.m["birads_encoded"].fillna(0).astype(int).to_numpy()
        self.density = self.m["density_encoded"].fillna(0).astype(int).to_numpy()
        self.birads_mask = self.m["birads_valid"].astype(bool).to_numpy()
        self.density_mask = self.m["density_valid"].astype(bool).to_numpy()

    def __len__(self) -> int:
        return len(self.m)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.m.iloc[idx]
        views = [self.pre.load_letterboxed(Path(r["cc_image_path"])), self.pre.load_letterboxed(Path(r["mlo_image_path"]))]
        if self.aug is not None:
            views = self.aug(views)
        images = torch.stack([self.pre.finalize(v) for v in views])
        item = {
            "images": images,
            "birads": torch.tensor(self.birads[idx], dtype=torch.long),
            "density": torch.tensor(self.density[idx], dtype=torch.long),
            "birads_mask": torch.tensor(bool(self.birads_mask[idx])),
            "density_mask": torch.tensor(bool(self.density_mask[idx])),
            "sample_id": str(r["sample_id"]),
        }
        if self.age is not None:
            item["age"] = self.age[idx]
        return item


def age_features_for(manifest: pd.DataFrame, processor) -> np.ndarray:
    """Transform manifest ages with a processor fitted on Train only."""
    return processor.transform(manifest["age"].to_numpy(dtype=float))
