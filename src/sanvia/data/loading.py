"""CSV loading, required-column validation and schema reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from sanvia.utils.paths import ProjectPaths

logger = logging.getLogger(__name__)


def read_csv_checked(path: Path, name: str) -> pd.DataFrame:
    """Read a CSV and fail with a diagnostic message if it is missing or empty."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"{name} is empty: {path}")
    logger.info("Loaded %s: shape=%s", name, df.shape)
    return df


def load_tables(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    return {
        "metadata": read_csv_checked(paths.metadata_csv, "metadata.csv"),
        "breast": read_csv_checked(paths.breast_annotations_csv, "breast-level_annotations.csv"),
        "finding": read_csv_checked(paths.finding_annotations_csv, "finding_annotations.csv"),
    }


def validate_required_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns {missing}. Available columns: {list(df.columns)}")


def normalize_id(value: Any, extension: str = ".png") -> str:
    """Strip whitespace and one trailing image extension; no substring matching."""
    text = str(value).strip()
    if text.lower().endswith(extension.lower()):
        text = text[: -len(extension)]
    return text


def build_schema_report(
    tables: dict[str, pd.DataFrame], paths: dict[str, Path], max_unique: int = 30, sample_rows: int = 5
) -> dict[str, Any]:
    """Shape, columns, dtypes, missing values, duplicate rows, categorical values and a sample for each CSV."""
    report: dict[str, Any] = {}
    for name, df in tables.items():
        categorical = {
            col: sorted(map(str, df[col].dropna().unique().tolist()))
            for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) <= max_unique
        }
        report[name] = {
            "path": str(paths[name]),
            "exists": Path(paths[name]).exists(),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
            "duplicate_rows": int(df.duplicated().sum()),
            "categorical_unique_values": categorical,
            "sample_records": df.sample(min(sample_rows, len(df)), random_state=0).to_dict(orient="records"),
        }
    return report
