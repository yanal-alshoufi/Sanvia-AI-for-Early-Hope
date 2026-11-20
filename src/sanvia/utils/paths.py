"""Resolved project paths."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectPaths:
    data_dir: Path
    images_dir: Path
    metadata_csv: Path
    breast_annotations_csv: Path
    finding_annotations_csv: Path
    output_dir: Path
    audit_dir: Path
    manifests_dir: Path
    splits_dir: Path
    preprocessing_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    plots_dir: Path
    reports_dir: Path
    mlops_dir: Path
    mlflow_dir: Path
    mlflow_db: Path
    mlflow_artifacts: Path
    dvc_remote_dir: Path

    def output_directories(self) -> list[Path]:
        names = (
            "audit_dir",
            "manifests_dir",
            "splits_dir",
            "preprocessing_dir",
            "checkpoints_dir",
            "metrics_dir",
            "plots_dir",
            "reports_dir",
            "mlops_dir",
            "mlflow_dir",
            "mlflow_artifacts",
            "dvc_remote_dir",
        )
        return [getattr(self, n) for n in names]

    def as_dict(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}


def build_paths(cfg: Mapping[str, Any]) -> ProjectPaths:
    p = cfg["data"]["paths"]
    data_dir = Path(p["data_dir"])
    out = data_dir / p["output_dir"]
    ml = cfg["mlops"]
    return ProjectPaths(
        data_dir=data_dir,
        images_dir=data_dir / p["images_dir"],
        metadata_csv=data_dir / p["metadata_csv"],
        breast_annotations_csv=data_dir / p["breast_annotations_csv"],
        finding_annotations_csv=data_dir / p["finding_annotations_csv"],
        output_dir=out,
        audit_dir=out / "data_audit",
        manifests_dir=out / "manifests",
        splits_dir=out / "splits",
        preprocessing_dir=out / "preprocessing",
        checkpoints_dir=out / "checkpoints",
        metrics_dir=out / "metrics",
        plots_dir=out / "plots",
        reports_dir=out / "reports",
        mlops_dir=out / "mlops",
        mlflow_dir=(out / ml["mlflow"]["db_relpath"]).parent,
        mlflow_db=out / ml["mlflow"]["db_relpath"],
        mlflow_artifacts=out / ml["mlflow"]["artifacts_relpath"],
        dvc_remote_dir=out / ml["dvc"]["remote_relpath"],
    )


def verify_input_paths(paths: ProjectPaths) -> None:
    """Raise a clear error naming every missing required input path."""
    required = {
        "DATA_DIR": paths.data_dir,
        "IMAGES_DIR": paths.images_dir,
        "METADATA_CSV": paths.metadata_csv,
        "BREAST_ANNOTATIONS_CSV": paths.breast_annotations_csv,
        "FINDING_ANNOTATIONS_CSV": paths.finding_annotations_csv,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required dataset paths are missing (is Google Drive mounted?):\n- " + "\n- ".join(missing))


def ensure_output_dirs(paths: ProjectPaths) -> None:
    for directory in paths.output_directories():
        directory.mkdir(parents=True, exist_ok=True)
