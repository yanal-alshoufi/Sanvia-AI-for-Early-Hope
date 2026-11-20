import math

import numpy as np
import pytest
import torch

from sanvia.mlops.reproducibility import find_resumable_run, update_run_meta, write_run_meta
from sanvia.training.checkpointing import load_checkpoint, save_checkpoint, verify_resume_compatibility
from sanvia.training.early_stopping import EarlyStopping
from sanvia.training.scheduler import build_warmup_cosine


def test_checkpoint_roundtrip_atomic(tmp_path):
    model = torch.nn.Linear(3, 2)
    size = save_checkpoint(tmp_path / "c.pt", {"model_state_dict": model.state_dict(), "epoch": 3})
    ck = load_checkpoint(tmp_path / "c.pt")
    assert size > 0 and ck["epoch"] == 3 and not list(tmp_path.glob("*.tmp"))
    fresh = torch.nn.Linear(3, 2)
    fresh.load_state_dict(ck["model_state_dict"])
    assert torch.equal(fresh.weight, model.weight)


def test_resume_validation_blocks_incompatible_config():
    ck = {"config_hash": "a", "dataset_id": "d1", "split_id": "s"}
    verify_resume_compatibility(ck, {"config_hash": "a", "dataset_id": "d1"})
    with pytest.raises(RuntimeError, match="incompatible"):
        verify_resume_compatibility(ck, {"dataset_id": "d2"})


def test_early_stopping_state_and_nan():
    es = EarlyStopping(2)
    assert es.step(0.5) and not es.step(0.4) and not es.should_stop
    assert not es.step(math.nan) and es.should_stop
    es2 = EarlyStopping(2)
    es2.load_state_dict(es.state_dict())
    assert es2.should_stop and es2.best == 0.5


def test_warmup_cosine_schedule():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = build_warmup_cosine(opt, total_steps=10, warmup_steps=2, min_lr_ratio=0.1)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs[0] < lrs[1] <= 1.0 and lrs[-1] < lrs[2] and min(lrs) >= 0.0


def test_find_resumable_run_matches_hashes(tmp_path):
    d = tmp_path / "ResNet50" / "run1"
    d.mkdir(parents=True)
    write_run_meta(d, {"status": "running", "config_hash": "c", "dataset_id": "d", "split_id": "s"})
    (d / "last.pt").write_bytes(b"x")
    assert find_resumable_run(tmp_path, "ResNet50", "c", "d", "s") == d
    assert find_resumable_run(tmp_path, "ResNet50", "other", "d", "s") is None
    update_run_meta(d, status="completed")
    assert find_resumable_run(tmp_path, "ResNet50", "c", "d", "s") is None


@pytest.mark.parametrize("use_mlflow", [False, True])
def test_end_to_end_training_and_resume(prepared, use_mlflow):
    from sanvia.mlops.mlflow_tracking import setup_mlflow
    from sanvia.training.runner import train_backbone

    cfg, paths, td = prepared["cfg"], prepared["paths"], prepared["td"]
    git = {"commit": "test", "dirty": False}
    env = {"gpu": {"gpu_name": None}}
    if use_mlflow:
        setup_mlflow(cfg, paths)
    s = train_backbone(
        "MobileNetV3-Large", cfg, paths, td, git, env, use_mlflow=use_mlflow, pretrained=False, device=torch.device("cpu")
    )
    run_dir = paths.checkpoints_dir / "MobileNetV3-Large" / s["run_id"]
    for f in (
        "best.pt",
        "last.pt",
        "run_summary.json",
        "history.csv",
        "confusion_birads.png",
        "learning_curves.png",
        "resolved_config.yaml",
    ):
        assert (run_dir / f).exists(), f
    ck = load_checkpoint(run_dir / "best.pt")
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_score",
        "backbone_name",
        "weights_identifier",
        "model_config",
        "preprocessing_config",
        "class_weights",
        "seed",
        "split_id",
        "dataset_manifest_hash",
    ):
        assert key in ck, key
    assert 0.0 <= s["validation"]["selection_score"] <= 1.0 and s["validation"]["n_samples"] == len(td.val_df)
    assert not any("official_test" in str(x) for x in td.val_df["final_split"].unique())
    assert (s["mlflow_run_id"] is not None) == use_mlflow
    # a second run never overwrites the first
    s2 = train_backbone(
        "MobileNetV3-Large", cfg, paths, td, git, env, use_mlflow=False, pretrained=False, device=torch.device("cpu")
    )
    assert s2["run_dir"] != s["run_dir"] and (run_dir / "best.pt").exists()


def test_interrupted_run_resumes_without_restarting(prepared):
    from sanvia.mlops.reproducibility import update_run_meta
    from sanvia.training.runner import train_backbone

    cfg, paths, td = prepared["cfg"], prepared["paths"], prepared["td"]
    git, env = {"commit": "t", "dirty": False}, {"gpu": {"gpu_name": None}}
    cfg["training"]["phase2"]["epochs"] = 3
    s = train_backbone(
        "MobileNetV3-Large", cfg, paths, td, git, env, use_mlflow=False, pretrained=False, device=torch.device("cpu")
    )
    run_dir = paths.checkpoints_dir / "MobileNetV3-Large" / s["run_id"]
    ck = load_checkpoint(run_dir / "last.pt")
    ck["epoch"], ck["history"], ck["phase"] = 2, ck["history"][:2], 2  # emulate interruption after epoch 2
    save_checkpoint(run_dir / "last.pt", ck)
    update_run_meta(run_dir, status="running")
    s2 = train_backbone(
        "MobileNetV3-Large", cfg, paths, td, git, env, resume=True, use_mlflow=False, pretrained=False, device=torch.device("cpu")
    )
    assert s2["run_id"] == s["run_id"] and s2["epochs_run"] == 4
    cfg["training"]["phase2"]["epochs"] = 2
    s3 = train_backbone(
        "MobileNetV3-Large", cfg, paths, td, git, env, resume=True, use_mlflow=False, pretrained=False, device=torch.device("cpu")
    )
    assert s3["run_id"] != s["run_id"]  # changed configuration forces a new run
