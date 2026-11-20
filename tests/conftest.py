import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[1]

from sanvia.utils.config import deep_update, load_config  # noqa: E402
from sanvia.utils.paths import build_paths, ensure_output_dirs  # noqa: E402
from synthetic import make_synthetic_dataset, shrink_backbones, synthetic_overrides  # noqa: E402


@pytest.fixture(scope="session")
def base_cfg():
    return load_config(ROOT / "configs")


@pytest.fixture()
def synth(tmp_path, base_cfg):
    make_synthetic_dataset(tmp_path)
    cfg = load_config(ROOT / "configs", synthetic_overrides(tmp_path))
    shrink_backbones(cfg)
    paths = build_paths(cfg)
    ensure_output_dirs(paths)
    return cfg, paths


@pytest.fixture()
def prepared(synth):
    """Full data preparation on the synthetic dataset (audit, split, samples, training data)."""
    from sanvia.data import pipeline as pl

    cfg, paths = synth
    index, manifest, _ = pl.index_and_manifest(cfg, paths)
    audit = pl.audit_stage(cfg, paths, index, manifest)
    split = pl.split_stage(cfg, paths, audit)
    samples = pl.sample_stage(cfg, paths, audit, split)
    td = pl.make_training_data(
        cfg,
        paths,
        samples["samples"],
        audit,
        samples["age_enabled"],
        samples["age_reason"],
        samples["dataset_record"],
        split["split_id"],
    )
    return {
        "cfg": cfg,
        "paths": paths,
        "index": index,
        "manifest": manifest,
        "audit": audit,
        "split": split,
        "samples": samples,
        "td": td,
    }
