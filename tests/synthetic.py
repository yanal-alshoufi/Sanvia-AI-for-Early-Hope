"""Synthetic VinDr-like dataset (grayscale PNGs and CSVs) for tests and smoke runs. No real medical data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

BIRADS = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]
DENSITY = ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]


def write_png(path: Path, rng: np.random.RandomState, size: tuple[int, int] = (80, 64), bit_depth: int = 8) -> None:
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((yy - h / 2) / (h / 3)) ** 2 + ((xx - w / 3) / (w / 3)) ** 2))
    img = np.clip(blob * 0.8 + rng.rand(h, w) * 0.1, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if bit_depth == 16:
        Image.fromarray((img * 65535).astype(np.uint16)).save(path)
    else:
        Image.fromarray((img * 255).astype(np.uint8)).save(path)


def make_synthetic_dataset(
    root: Path, n_studies: int = 40, n_train_official: int = 30, bit_depth: int = 8, with_age: bool = True, seed: int = 0
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    images_dir = root / "images" / "images_png"
    rows, meta_rows, find_rows = [], [], []
    for s in range(n_studies):
        study = f"study{s:03d}"
        split = "training" if s < n_train_official else "test"
        age = int(rng.randint(30, 80))
        for lat in ("L", "R"):
            b, d = BIRADS[(s + (lat == "R")) % 5], DENSITY[(s * 2 + (lat == "R")) % 4]
            for view in ("CC", "MLO"):
                image_id = f"img{s:03d}{lat}{view}"
                write_png(images_dir / study / f"{image_id}.png", rng, bit_depth=bit_depth)
                rows.append(
                    {
                        "study_id": study,
                        "series_id": f"ser{s:03d}{lat}{view}",
                        "image_id": image_id,
                        "laterality": lat,
                        "view_position": view,
                        "height": 80,
                        "width": 64,
                        "breast_birads": b,
                        "breast_density": d,
                        "split": split,
                    }
                )
                meta_rows.append({"SOP Instance UID": image_id, "Patient's Age": f"0{age}Y", "Manufacturer": "synthetic"})
                if s % 5 == 0 and view == "CC":
                    find_rows.append(
                        {
                            "study_id": study,
                            "series_id": f"ser{s:03d}{lat}{view}",
                            "image_id": image_id,
                            "laterality": lat,
                            "view_position": view,
                            "height": 80,
                            "width": 64,
                            "breast_birads": b,
                            "breast_density": d,
                            "finding_categories": "['Mass']",
                            "finding_birads": b,
                            "xmin": 5.0,
                            "ymin": 5.0,
                            "xmax": 40.0,
                            "ymax": 60.0,
                            "split": split,
                        }
                    )
    pd.DataFrame(rows).to_csv(root / "breast-level_annotations.csv", index=False)
    meta = pd.DataFrame(meta_rows)
    if not with_age:
        meta = meta.drop(columns=["Patient's Age"])
    meta.to_csv(root / "metadata.csv", index=False)
    pd.DataFrame(find_rows).to_csv(root / "finding_annotations.csv", index=False)
    return {"root": root, "images_dir": images_dir}


def synthetic_overrides(root: Path, tiny_sizes: bool = True) -> dict[str, Any]:
    """Config overrides pointing at the synthetic dataset with a tiny, fast training protocol."""
    ov: dict[str, Any] = {
        "data": {
            "paths": {"data_dir": str(root)},
            "audit": {"stats_sample_size": None, "workers": 2},
            "indexing": {"hash_workers": 2},
            "split": {"min_stratum_size": 2},
        },
        "training": {
            "batch_size": 4,
            "effective_batch_size": 8,
            "num_workers": 0,
            "amp": False,
            "phase1": {"epochs": 1},
            "phase2": {"epochs": 2, "unfreeze_epochs": 2, "early_stopping_patience": 5},
            "inference_timing": {"batch_size": 2, "warmup_iters": 1, "iters": 2},
        },
    }
    return ov


def shrink_backbones(cfg: dict[str, Any], mobile_size: int = 64, other_size: int = 96) -> None:
    for b in cfg["backbones"]:
        b["input_size"] = mobile_size if b["name"] in {"MobileNetV3-Large"} else other_size
