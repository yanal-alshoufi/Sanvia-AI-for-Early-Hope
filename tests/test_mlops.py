import shutil
import subprocess

import pytest

from sanvia.mlops.data_versioning import DVCManager
from sanvia.mlops.git_metadata import get_git_metadata, reproducibility_statement
from sanvia.utils.config import validate_config
from sanvia.utils.io import sha256_file

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
needs_dvc = pytest.mark.skipif(shutil.which("dvc") is None, reason="dvc not installed")


def _git_repo(path):
    def run(*args):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)

    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (path / "a.txt").write_text("x")
    run("add", "a.txt")
    run("commit", "-qm", "init")


@needs_git
def test_git_metadata_clean_and_dirty(tmp_path):
    assert get_git_metadata(tmp_path)["available"] is False
    _git_repo(tmp_path)
    meta = get_git_metadata(tmp_path)
    assert meta["available"] and len(meta["commit"]) == 40 and meta["dirty"] is False
    (tmp_path / "a.txt").write_text("changed")
    dirty = get_git_metadata(tmp_path)
    assert dirty["dirty"] and "NOT fully reproducible" in reproducibility_statement(dirty)


def test_full_image_tracking_is_rejected(base_cfg):
    cfg = {**base_cfg, "mlops": {**base_cfg["mlops"], "dvc": {**base_cfg["mlops"]["dvc"], "track_full_images": True}}}
    with pytest.raises(ValueError, match="track_full_images"):
        validate_config(cfg)
    with pytest.raises(ValueError):
        DVCManager("/tmp", cfg, "/tmp/remote")


def test_official_test_evaluation_flag_is_rejected(base_cfg):
    cfg = {**base_cfg, "data": {**base_cfg["data"], "evaluate_official_test": True}}
    with pytest.raises(ValueError, match="evaluate_official_test"):
        validate_config(cfg)


@needs_git
@needs_dvc
def test_dvc_tracks_small_files_only_and_roundtrips(tmp_path, base_cfg):
    repo, remote = tmp_path / "repo", tmp_path / "remote"
    repo.mkdir()
    _git_repo(repo)
    csv = tmp_path / "metadata.csv"
    csv.write_text("a,b\n1,2\n")
    mgr = DVCManager(repo, base_cfg, remote)
    mgr.ensure_initialized()
    staged = mgr.stage_files({"metadata.csv": csv})
    assert (staged[0].parent / "metadata.csv.dvc").exists()
    png = tmp_path / "x.png"
    png.write_bytes(b"x")
    with pytest.raises(ValueError, match="images"):
        mgr.stage_files({"x.png": png})
    mgr.push()
    assert mgr.remote_in_sync()
    before = sha256_file(staged[0])
    assert mgr.verify_roundtrip(staged[0]) and sha256_file(staged[0]) == before
