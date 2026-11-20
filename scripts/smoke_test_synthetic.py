"""End-to-end check of the whole workflow on a synthetic dataset (no real data, no pretrained weights).

Runs data preparation, all six backbones (tiny protocol), Validation-only selection, MLflow registration and reports.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import torch  # noqa: E402

from sanvia.data import pipeline as pl  # noqa: E402
from sanvia.mlops.git_metadata import get_git_metadata  # noqa: E402
from sanvia.mlops.mlflow_tracking import setup_mlflow, verify_mlflow_persistence  # noqa: E402
from sanvia.training.benchmark import finalize_benchmark, run_all_backbones  # noqa: E402
from sanvia.utils.config import load_config  # noqa: E402
from sanvia.utils.environment import collect_environment  # noqa: E402
from sanvia.utils.logging import setup_logging  # noqa: E402
from sanvia.utils.paths import build_paths, ensure_output_dirs  # noqa: E402
from synthetic import make_synthetic_dataset, shrink_backbones, synthetic_overrides  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbones", nargs="*", default=None, help="Subset to run (default: all six). Use a subset on small disks."
    )
    parser.add_argument("--keep", action="store_true", help="Keep the temporary output directory.")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="sanvia_synth_"))
    make_synthetic_dataset(root, n_studies=60, n_train_official=45)
    cfg = load_config(ROOT / "configs", synthetic_overrides(root))
    shrink_backbones(cfg)
    if args.backbones:
        cfg["backbones"] = [b for b in cfg["backbones"] if b["name"] in args.backbones]
    cfg["training"]["phase2"]["epochs"] = 2
    setup_logging("INFO")
    paths = build_paths(cfg)
    ensure_output_dirs(paths)
    index, manifest, _ = pl.index_and_manifest(cfg, paths)
    audit = pl.audit_stage(cfg, paths, index, manifest)
    split = pl.split_stage(cfg, paths, audit)
    out = pl.sample_stage(cfg, paths, audit, split)
    assert not audit["critical_issues"] and not out["issues"]
    td = pl.make_training_data(
        cfg, paths, out["samples"], audit, out["age_enabled"], out["age_reason"], out["dataset_record"], split["split_id"]
    )
    setup_mlflow(cfg, paths)
    print(verify_mlflow_persistence(paths, cfg["mlops"]["mlflow"]["experiment_name"]))
    git, env = get_git_metadata(ROOT), collect_environment()
    summaries, failures = run_all_backbones(cfg, paths, td, git, env, pretrained=False, device=torch.device("cpu"))
    assert not failures, failures
    result = finalize_benchmark(cfg, paths, td, summaries, git, env)
    print(result["table"][["backbone", "birads_macro_f1", "density_macro_f1", "selection_score", "training_time_min"]])
    print("Selected:", result["selection"]["selected"])
    print("Registered:", result["registered"])
    print("Outputs:", paths.output_dir)
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
