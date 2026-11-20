"""Recursive PNG indexing and exact stem-based matching to annotation image ids."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from sanvia.data.loading import normalize_id

logger = logging.getLogger(__name__)


@dataclass
class ImageIndex:
    images_dir: Path
    stem_to_paths: dict[str, list[Path]] = field(default_factory=dict)

    @property
    def n_files(self) -> int:
        return sum(len(v) for v in self.stem_to_paths.values())

    @property
    def duplicates(self) -> dict[str, list[Path]]:
        return {k: v for k, v in self.stem_to_paths.items() if len(v) > 1}

    def unique_path(self, stem: str) -> Path | None:
        paths = self.stem_to_paths.get(stem, [])
        return paths[0] if len(paths) == 1 else None

    def relative(self, path: Path) -> str:
        return Path(path).relative_to(self.images_dir).as_posix()


def index_png_files(images_dir: Path, extension: str = ".png") -> ImageIndex:
    """Index all files with ``extension`` under ``images_dir`` recursively, keyed by filename stem."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"IMAGES_DIR does not exist: {images_dir}")
    index = ImageIndex(images_dir=images_dir)
    for root, _, files in os.walk(images_dir):
        for name in sorted(files):
            if name.lower().endswith(extension.lower()):
                path = Path(root) / name
                index.stem_to_paths.setdefault(name[: -len(extension)], []).append(path)
    logger.info(
        "Indexed %d PNG files (%d unique stems, %d duplicated stems)",
        index.n_files,
        len(index.stem_to_paths),
        len(index.duplicates),
    )
    return index


def match_annotations_to_images(image_ids: Iterable[str], index: ImageIndex, extension: str = ".png") -> pd.DataFrame:
    """Exact stem match for every distinct annotation image id: status is ok, missing or ambiguous."""
    rows = []
    for raw in pd.unique(pd.Series(list(image_ids), dtype=object)):
        stem = normalize_id(raw, extension)
        candidates = index.stem_to_paths.get(stem, [])
        status = "ok" if len(candidates) == 1 else ("missing" if not candidates else "ambiguous")
        rows.append(
            {
                "image_id": stem,
                "status": status,
                "n_candidates": len(candidates),
                "path": str(candidates[0]) if status == "ok" else None,
                "relative_path": index.relative(candidates[0]) if status == "ok" else None,
                "candidate_paths": ";".join(str(c) for c in candidates) if status == "ambiguous" else None,
            }
        )
    return pd.DataFrame(rows, columns=["image_id", "status", "n_candidates", "path", "relative_path", "candidate_paths"])


def find_unexpected_files(index: ImageIndex, referenced_stems: Iterable[str]) -> pd.DataFrame:
    referenced = set(referenced_stems)
    rows = [
        {"image_id": stem, "path": str(p)} for stem, paths in index.stem_to_paths.items() if stem not in referenced for p in paths
    ]
    return pd.DataFrame(rows, columns=["image_id", "path"])
