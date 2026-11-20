"""Age feature: join verification, study-level derivation and Train-only preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from sanvia.data.loading import normalize_id
from sanvia.utils.io import load_json, save_json

_AGE_PATTERN = re.compile(r"^\s*(\d{1,3})\s*[Yy]?\s*$")  # strict: digits with an optional trailing Y only


def parse_age(series: pd.Series) -> pd.Series:
    """Parse ages such as ``"057Y"`` or ``57``. Anything else becomes NaN."""

    def parse(value: Any) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        m = _AGE_PATTERN.match(str(value))
        return float(m.group(1)) if m else np.nan

    return series.map(parse).astype(float)


def verify_age_join(
    metadata: pd.DataFrame, breast: pd.DataFrame, age_cfg: Mapping[str, Any], image_col: str, study_col: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate the metadata-to-annotation identifier relationship and build per-image ages.

    Returns ``(report, image_age)`` where ``image_age`` has columns ``image_id`` and ``age``.
    """
    report: dict[str, Any] = {"age_column": age_cfg["metadata_age_column"], "candidates": {}, "verified": False}
    age_col = age_cfg["metadata_age_column"]
    if age_col not in metadata.columns:
        report["reason"] = f"Age column {age_col!r} not found in metadata.csv. Columns: {list(metadata.columns)}"
        return report, pd.DataFrame(columns=["image_id", "age"])
    ann_ids = pd.Series(breast[image_col].map(normalize_id).unique())
    best: tuple[str, float] | None = None
    for cand in age_cfg["metadata_id_candidates"]:
        if cand not in metadata.columns:
            report["candidates"][cand] = {"present": False}
            continue
        meta_ids = metadata[cand].map(normalize_id)
        rate = float(ann_ids.isin(set(meta_ids)).mean())
        report["candidates"][cand] = {
            "present": True,
            "exact_match_rate": rate,
            "duplicate_ids": int(meta_ids.duplicated().sum()),
            "unmatched_annotation_ids": int((~ann_ids.isin(set(meta_ids))).sum()),
        }
        if best is None or rate > best[1]:
            best = (cand, rate)
    if best is None or best[1] < age_cfg["min_match_rate"]:
        report["reason"] = (
            f"No identifier column reached the required exact match rate {age_cfg['min_match_rate']} (best: {best})."
        )
        return report, pd.DataFrame(columns=["image_id", "age"])
    id_col = best[0]
    meta = pd.DataFrame({"image_id": metadata[id_col].map(normalize_id), "raw_age": metadata[age_col]})
    meta["age"] = parse_age(meta["raw_age"])
    lo, hi = age_cfg["valid_range"]
    out_of_range = meta["age"].notna() & ~meta["age"].between(lo, hi)
    unparsable = meta["age"].isna() & meta["raw_age"].notna()
    meta.loc[out_of_range, "age"] = np.nan
    conflicts = meta.dropna(subset=["age"]).groupby("image_id")["age"].nunique()
    conflicting_ids = set(conflicts[conflicts > 1].index)
    per_image = meta.dropna(subset=["age"]).groupby("image_id")["age"].first().reset_index()
    per_image = per_image[~per_image["image_id"].isin(conflicting_ids)]
    report.update(
        id_column=id_col,
        match_rate=best[1],
        n_out_of_range=int(out_of_range.sum()),
        n_unparsable=int(unparsable.sum()),
        n_conflicting_image_ids=len(conflicting_ids),
        verified=True,
    )
    return report, per_image


def derive_study_age(breast: pd.DataFrame, image_age: pd.DataFrame, image_col: str, study_col: str) -> tuple[pd.DataFrame, dict]:
    """One age per study when all available image ages agree; conflicts and gaps become missing."""
    df = breast[[study_col, image_col]].copy()
    df["image_id"] = df[image_col].map(normalize_id)
    df = df.merge(image_age, on="image_id", how="left")
    rows = []
    for study, grp in df.groupby(study_col):
        values = sorted(set(grp["age"].dropna().tolist()))
        rows.append(
            {
                "study_id": study,
                "age": values[0] if len(values) == 1 else np.nan,
                "n_distinct_ages": len(values),
                "conflict": len(values) > 1,
            }
        )
    study_age = pd.DataFrame(rows)
    summary = {
        "n_studies": len(study_age),
        "n_with_age": int(study_age["age"].notna().sum()),
        "n_conflicting_studies": int(study_age["conflict"].sum()),
    }
    return study_age, summary


def decide_age_enabled(mode: str, report: Mapping[str, Any]) -> tuple[bool, str]:
    if mode == "disabled":
        return False, "Age branch disabled by configuration."
    if report.get("verified"):
        return True, f"Age verified via {report['id_column']} (match rate {report['match_rate']:.4f})."
    if mode == "enabled":
        raise RuntimeError(f"Age is required (mode=enabled) but could not be verified: {report.get('reason')}")
    return False, f"Age disabled: {report.get('reason')}"


@dataclass
class AgeProcessor:
    """Median imputation and standardisation fitted on Train only; emits ``[age_norm, missing_flag]``."""

    median: float | None = None
    mean: float | None = None
    std: float | None = None

    def fit(self, train_ages: np.ndarray) -> "AgeProcessor":
        ages = np.asarray(train_ages, dtype=float)
        valid = ages[~np.isnan(ages)]
        if valid.size == 0:
            raise ValueError("Cannot fit the age processor: no valid ages in Train.")
        self.median = float(np.median(valid))
        filled = np.where(np.isnan(ages), self.median, ages)
        self.mean = float(filled.mean())
        self.std = float(max(filled.std(), 1e-6))
        return self

    def transform(self, ages: np.ndarray) -> np.ndarray:
        if self.median is None:
            raise RuntimeError("AgeProcessor must be fitted before transform.")
        ages = np.asarray(ages, dtype=float)
        flag = np.isnan(ages).astype(np.float32)
        filled = np.where(np.isnan(ages), self.median, ages)
        return np.stack([((filled - self.mean) / self.std).astype(np.float32), flag], axis=1)

    def to_dict(self) -> dict[str, float | None]:
        return {"median": self.median, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AgeProcessor":
        return cls(d["median"], d["mean"], d["std"])

    def save(self, path: Path) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "AgeProcessor":
        return cls.from_dict(load_json(path))
