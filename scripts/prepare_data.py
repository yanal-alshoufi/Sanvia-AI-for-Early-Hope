"""Run the data stages (manifests, audit, split, samples) from the command line.

Usage: python scripts/prepare_data.py [--config-dir configs] [--data-dir /path/to/VnDir_Mammo]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sanvia.data import pipeline as pl  # noqa: E402
from sanvia.utils.config import load_config  # noqa: E402
from sanvia.utils.logging import setup_logging  # noqa: E402
from sanvia.utils.paths import build_paths, ensure_output_dirs, verify_input_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    overrides = {"data": {"paths": {"data_dir": args.data_dir}}} if args.data_dir else None
    cfg = load_config(args.config_dir, overrides)
    setup_logging(cfg["logging"]["level"])
    paths = build_paths(cfg)
    verify_input_paths(paths)
    ensure_output_dirs(paths)
    index, manifest, _ = pl.index_and_manifest(cfg, paths)
    audit = pl.audit_stage(cfg, paths, index, manifest)
    if audit["critical_issues"]:
        raise SystemExit(f"Critical data issues: {audit['critical_issues']}")
    split = pl.split_stage(cfg, paths, audit)
    out = pl.sample_stage(cfg, paths, audit, split)
    if out["issues"]:
        raise SystemExit(f"Critical sample issues: {out['issues']}")
    print(f"Prepared {len(out['samples'])} breast samples; split id {split['split_id']}")


if __name__ == "__main__":
    main()
