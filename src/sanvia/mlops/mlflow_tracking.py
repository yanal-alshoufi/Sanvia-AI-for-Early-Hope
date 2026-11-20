"""MLflow tracking on a persistent SQLite backend and Drive artifact store (single-user, sequential use)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping

import mlflow

from sanvia.utils.paths import ProjectPaths

logger = logging.getLogger(__name__)


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def setup_mlflow(cfg: Mapping[str, Any], paths: ProjectPaths) -> str:
    """Point MLflow at the persistent database and artifact directory and create the experiment."""
    paths.mlflow_dir.mkdir(parents=True, exist_ok=True)
    paths.mlflow_artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{paths.mlflow_db.as_posix()}")
    name = cfg["mlops"]["mlflow"]["experiment_name"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, artifact_location=paths.mlflow_artifacts.as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(name)
    return experiment_id


def verify_mlflow_persistence(paths: ProjectPaths, experiment_name: str) -> dict[str, Any]:
    """Log a test run, read it back and confirm that the database and artifact directory live on persistent storage."""
    with mlflow.start_run(run_name="persistence_check") as run:
        mlflow.log_param("check", "ok")
        mlflow.log_metric("check_metric", 1.0)
        mlflow.set_tag("purpose", "persistence_check")
        run_id = run.info.run_id
    stored = mlflow.get_run(run_id)
    return {
        "db_exists": paths.mlflow_db.exists(),
        "db_path": str(paths.mlflow_db),
        "artifact_dir_exists": paths.mlflow_artifacts.exists(),
        "artifact_dir": str(paths.mlflow_artifacts),
        "readback_ok": stored.data.params.get("check") == "ok",
        "run_id": run_id,
        "experiment": experiment_name,
    }


class NullTracker:
    """No-op tracker used for tests and runs without MLflow."""

    run_id: str | None = None

    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None: ...
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None: ...
    def log_dict(self, obj: Any, artifact_file: str) -> None: ...
    def set_tags(self, tags: Mapping[str, Any]) -> None: ...
    def end(self, status: str = "FINISHED") -> None: ...


class MLflowTracker(NullTracker):
    """One MLflow run per backbone experiment."""

    def __init__(self, run_name: str, tags: Mapping[str, Any] | None = None, existing_run_id: str | None = None):
        run = mlflow.start_run(run_id=existing_run_id) if existing_run_id else mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id
        if tags and not existing_run_id:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_params(self, params: Mapping[str, Any]) -> None:
        flat = {k: (str(v)[:500]) for k, v in flatten(params).items()}
        items = list(flat.items())
        for i in range(0, len(items), 90):
            mlflow.log_params(dict(items[i : i + 90]))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and not math.isnan(float(v))}
        if clean:
            mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if Path(path).exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, obj: Any, artifact_file: str) -> None:
        mlflow.log_dict(obj, artifact_file)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def end(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)
