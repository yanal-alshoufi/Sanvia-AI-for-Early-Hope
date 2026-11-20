"""Label spaces and exact-match label encoding (no regex extraction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


@dataclass(frozen=True)
class LabelSpace:
    name: str
    mapping: dict[str, int]  # upper-cased original string -> encoded index
    num_classes: int
    class_names: tuple[str, ...]


def build_label_spaces(cfg: Mapping[str, Any]) -> tuple[LabelSpace, LabelSpace]:
    labels = cfg["data"]["labels"]

    def make(name: str, raw: Mapping[str, int], names: tuple[str, ...]) -> LabelSpace:
        mapping = {str(k).strip().upper(): int(v) for k, v in raw.items()}
        return LabelSpace(name, mapping, len(names), names)

    birads = make("birads", labels["birads_map"], tuple(f"BI-RADS {i}" for i in range(1, 6)))
    density = make("density", labels["density_map"], tuple(f"Density {c}" for c in "ABCD"))
    return birads, density


def is_missing(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return True
    return str(value).strip().lower() in MISSING_TOKENS


def encode_value(value: Any, space: LabelSpace) -> tuple[int | None, str]:
    """Return ``(encoded, status)`` where status is ``ok``, ``missing`` or ``unknown``."""
    if is_missing(value):
        return None, "missing"
    key = str(value).strip().upper()
    if key in space.mapping:
        return space.mapping[key], "ok"
    return None, "unknown"


def encode_series(series: pd.Series, space: LabelSpace) -> tuple[pd.Series, pd.Series]:
    """Encode a label column. Returns a nullable-integer series and a status series."""
    results = [encode_value(v, space) for v in series]
    encoded = pd.Series([r[0] for r in results], index=series.index, dtype="Int64")
    status = pd.Series([r[1] for r in results], index=series.index, dtype="object")
    return encoded, status
