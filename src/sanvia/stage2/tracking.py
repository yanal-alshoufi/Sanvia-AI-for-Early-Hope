"""Single-writer local MLflow tracking and narrowly scoped DVC integration."""

from __future__ import annotations

import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np

from sanvia.mlops.data_versioning import DVCManager
from sanvia.mlops.mlflow_tracking import MLflowTracker, NullTracker, setup_mlflow
from sanvia.utils.io import load_json, save_json
from sanvia.utils.paths import build_paths

LOG = logging.getLogger(__name__)
EXPERIMENT = "SANVIA_Stage2_Domain_Adaptation"


def paths_for(output: Path):
    ml = dict(
        mlflow=dict(experiment_name=EXPERIMENT, db_relpath="tracking/mlflow.db", artifacts_relpath="tracking/artifacts"),
        dvc=dict(
            track_full_images=False,
            remote_name="sanvia_stage2_storage",
            remote_relpath="tracking/dvc_remote",
            tracked_dir="data/stage2_tracked",
        ),
    )
    cfg = dict(
        mlops=ml,
        data=dict(
            paths=dict(
                data_dir=str(output),
                output_dir=".",
                images_dir=".",
                metadata_csv="",
                breast_annotations_csv="",
                finding_annotations_csv="",
            )
        ),
    )
    return cfg, build_paths(cfg)


class ReportingTracker(NullTracker):
    """Keep local checkpoints authoritative if a logging backend fails."""

    def __init__(self, tracker, path):
        self.tracker, self.path = tracker, path
        self.run_id = tracker.run_id

    def call(self, method, *args):
        try:
            return getattr(self.tracker, method)(*args)
        except Exception as exc:
            LOG.error("MLflow %s failed: %s", method, exc)
            save_json(dict(action=method, error=str(exc)), self.path / "mlflow_failure.json")

    def log_metrics(self, metrics, step):
        self.call("log_metrics", metrics, step)


@contextmanager
def tracked(output, folder, identity, enabled):
    folder.mkdir(parents=True, exist_ok=True)
    tracker = ReportingTracker(NullTracker(), folder)
    if enabled:
        try:
            cfg, paths = paths_for(output)
            experiment = setup_mlflow(cfg, paths)
            parent_path = folder.parent / "mlflow_parent.json"
            client = mlflow.tracking.MlflowClient()
            if parent_path.exists():
                parent_id = load_json(parent_path)["run_id"]
            else:
                parent = client.create_run(
                    experiment,
                    tags={
                        "mlflow.runName": f"{identity['backbone']}-{identity['strategy']}-{identity['seed']}",
                        "benchmark": identity["benchmark"],
                    },
                )
                parent_id = parent.info.run_id
                save_json({"run_id": parent_id}, parent_path)
                client.set_terminated(parent_id)
            meta = folder / "mlflow_run.json"
            previous = load_json(meta)["run_id"] if meta.exists() else None
            live = MLflowTracker(
                f"{identity['backbone']}-{identity['strategy']}-{identity['seed']}-{folder.name}",
                dict(identity, **{"mlflow.parentRunId": parent_id}),
                previous,
            )
            tracker = ReportingTracker(live, folder)
            save_json(dict(run_id=live.run_id), meta)
            tracker.call("log_dict", identity, "identity.json")
            tracker.call("log_artifact", output / "reproducibility_report.json", "protocol")
        except Exception as exc:
            LOG.error("MLflow initialization failed: %s", exc)
            save_json(dict(action="start", error=str(exc)), folder / "mlflow_failure.json")
    try:
        yield tracker
    except BaseException:
        tracker.call("end", "FAILED")
        raise
    else:
        if tracker.run_id:
            for path in folder.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix in {".json", ".csv", ".png", ".pt", ".yaml"}
                    and path.name not in {"last_checkpoint.pt", "best.pt"}
                ):
                    tracker.call("log_artifact", path, str(path.parent.relative_to(folder)) if path.parent != folder else None)
            tracker.call("end", "FINISHED")


def stage_dvc(repo: Path, out: Path, files: dict):
    """Track authorized compact definitions and hash references in a per-benchmark namespace."""
    cfg, paths = paths_for(out)
    cfg["mlops"]["dvc"]["tracked_dir"] += f"/{out.name}"
    workspace = out / "tracking" / "dvc_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    manager = DVCManager(workspace, cfg, paths.dvc_remote_dir)
    manager.ensure_initialized()
    staged = manager.stage_files(files)
    manager.push()
    result = dict(
        files=[str(p.relative_to(workspace)) for p in staged], remote_status=manager.status_cloud(), images_backed_up=False
    )
    save_json(result, out / "dvc_status.json")
    return result


def register_handoff(handoff: Path, output: Path):
    """Register research-only backbone artifacts in the benchmark's persistent local registry."""
    from sanvia.stage2.export import load_export

    cfg, paths = paths_for(output)
    setup_mlflow(cfg, paths)
    client = mlflow.tracking.MlflowClient()
    for name, choice in load_json(handoff / "selected_initializations.json").items():
        registered = f"SANVIA_Stage2_{name}"
        versions = client.search_model_versions(f"name='{registered}'")
        existing = next((v for v in versions if v.tags.get("export_sha256") == choice["sha256"]), None)
        if existing is not None:
            client.set_registered_model_alias(registered, "research_candidate", existing.version)
            continue
        with mlflow.start_run(run_name=f"research-handoff-{name}"):
            model = load_export(handoff / choice["file"])
            size = model.spec.input_size
            example = np.zeros((2, 3, size, size), dtype=np.float32)
            info = mlflow.pytorch.log_model(model.eval(), artifact_path="backbone", input_example=example)
            version = mlflow.register_model(info.model_uri, registered)
            client.set_model_version_tag(registered, version.version, "export_sha256", choice["sha256"])
            client.set_registered_model_alias(registered, "research_candidate", version.version)
