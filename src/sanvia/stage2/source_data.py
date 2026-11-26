"""Lazy fixed-slot external datasets with deterministic per-draw training augmentation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sanvia.data.dataset import PairedAugmenter
from sanvia.data.preprocessing import imagenet_normalize, replicate_channels
from sanvia.stage2.preprocessing import preprocess
from sanvia.stage2.stage1_artifacts import local_path, require
from sanvia.training.losses import compute_class_weights
from sanvia.utils.seed import make_generator


class SourceDataset(Dataset):
    def __init__(self, data: dict, root: Path, split: str, size: int, pre: dict, aug: dict, cache: Path | None):
        require(split in {"train", "validation"}, "Only source train and validation datasets may be constructed")
        self.rows = [s for s in data["samples"] if s["eligible"] and s["split"] == split]
        require(self.rows and all(s["official_split"] != "test" for s in self.rows), f"Invalid source {split} partition")
        self.tasks = data["tasks"]
        self.index = data["images"]
        self.root, self.size, self.pre, self.cache = root, size, pre, cache
        self.aug = PairedAugmenter(aug) if split == "train" and aug["enabled"] else None
        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        draw = 0
        if isinstance(idx, tuple):
            idx, draw = idx
        row = self.rows[idx]
        positions = [i for i, v in enumerate(row["views"]) if v is not None]
        tensors = []
        for i in positions:
            info = self.index[row["views"][i]]
            tensors.append(preprocess(local_path(self.root, info["image_path"]), info["sha256"], self.size, self.pre, self.cache))
        if self.aug:
            state = random.getstate()
            try:
                with torch.random.fork_rng(devices=[]):
                    random.seed(draw)
                    torch.default_generator.manual_seed(draw)
                    tensors = self.aug(tensors)
            finally:
                random.setstate(state)
        views = torch.zeros(2, 3, self.size, self.size)
        mask = torch.zeros(2, dtype=torch.bool)
        for i, tensor in zip(positions, tensors):
            mask[i] = True
            views[i] = imagenet_normalize(replicate_channels(tensor), self.pre["imagenet_mean"], self.pre["imagenet_std"])
        item = dict(images=views, view_mask=mask, sample_id=row["sample_id"])
        for k, t in self.tasks.items():
            values = row["tasks"][k]
            for out, field in [(k, "y"), (k + "_mask", "mask")]:
                v = [a[field] for a in values]
                item[out] = torch.as_tensor(v[0] if t["scope"] == "breast" else v)
        return item

    def weights(self, device) -> dict:
        require(self.split == "train", "Class weights require source training data")
        result = {}
        for k, t in self.tasks.items():
            if t["loss"] == "ce":
                labels = [v["y"] for row in self.rows for v in row["tasks"][k] if v["mask"]]
                result[k] = torch.as_tensor(
                    compute_class_weights(np.array(labels, dtype=int), len(t["classes"])), device=device, dtype=torch.float32
                )
        return result


def loader(dataset, settings, indices=None, seed=0):
    workers = settings["workers"]
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        sampler=indices,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=settings["prefetch_factor"] if workers else None,
        pin_memory=settings["pin_memory"] and torch.cuda.is_available(),
        generator=make_generator(seed),
    )


def draws(size: int, start: int, count: int, seed: int):
    """Immutable position in a sequence of independently shuffled training passes."""
    cycle = -1
    for pos in range(start, start + count):
        if pos // size != cycle:
            cycle = pos // size
            order = np.random.default_rng([seed, cycle]).permutation(size)
        yield int(order[pos % size]), (seed * 1000003 + pos) % (2**63)
