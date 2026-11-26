"""Run and record the actual checkout regression suite before real training."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sanvia.utils.config import config_hash
from sanvia.utils.io import load_json, save_json, sha256_file


def preflight(repo: Path, out: Path, code: str) -> dict:
    tests = repo / "tests"
    key = config_hash(dict(code=code, tests={str(p.relative_to(repo)): sha256_file(p) for p in tests.rglob("*.py")}), 64)
    marker = out / "preflight.json"
    if marker.is_file():
        result = load_json(marker)
        if result.get("key") == key and result.get("passed"):
            return result
    command = [sys.executable, "-m", "pytest", "-q", str(tests)]
    with (out / "preflight.log").open("w") as log:
        run = subprocess.run(
            command,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        )
    result = dict(key=key, command=command, passed=run.returncode == 0, log="preflight.log")
    save_json(result, marker)
    if run.returncode:
        raise RuntimeError(f"Preflight failed. Inspect {out / 'preflight.log'} before training.")
    return result
