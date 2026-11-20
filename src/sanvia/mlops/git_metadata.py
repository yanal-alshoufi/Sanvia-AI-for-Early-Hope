"""Git commit and repository-state capture."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def get_git_metadata(repo_root: Path) -> dict[str, Any]:
    """Return commit hash, branch and dirty flag; ``available=False`` outside a Git repository."""
    try:
        status = _git(repo_root, "status", "--porcelain")
        return {
            "available": True,
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "uncommitted_files": [line[3:] for line in status.splitlines()],
            "remote": _safe_remote(repo_root),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "available": False,
            "commit": "unavailable",
            "branch": None,
            "dirty": None,
            "uncommitted_files": [],
            "remote": None,
        }


def _safe_remote(repo_root: Path) -> str | None:
    try:
        return _git(repo_root, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        return None


def reproducibility_statement(meta: dict[str, Any]) -> str:
    if not meta["available"]:
        return "Git metadata unavailable: code state is not traceable."
    if meta["dirty"]:
        return "Repository has uncommitted changes: the run is NOT fully reproducible from the commit hash alone."
    return "Repository clean: code is reproducible from the recorded commit hash."
