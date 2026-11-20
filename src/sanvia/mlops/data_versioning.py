"""PNG manifests with SHA-256 checksums, dataset snapshot ids and DVC helpers.

DVC versions only small files (CSVs, manifests, splits). PNG images are verified through
the checksum manifest only; the manifest detects changes but cannot restore old image versions.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sanvia.data.image_indexing import ImageIndex
from sanvia.data.png_io import PNGFormatError, read_png_header
from sanvia.utils.io import load_json, save_json, sha256_file, sha256_text

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "image_id",
    "relative_path",
    "file_size",
    "mtime_ns",
    "sha256",
    "width",
    "height",
    "bit_depth",
    "color_type",
    "header_error",
]


def _manifest_row(stem: str, path: Path, images_dir: Path, previous: dict | None) -> dict[str, Any]:
    stat = path.stat()
    rel = path.relative_to(images_dir).as_posix()
    row: dict[str, Any] = {
        "image_id": stem,
        "relative_path": rel,
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "header_error": None,
    }
    prev = previous.get(rel) if previous else None
    if prev and prev["file_size"] == stat.st_size and prev["mtime_ns"] == stat.st_mtime_ns:
        row.update({k: prev[k] for k in ("sha256", "width", "height", "bit_depth", "color_type")})
        return row
    row["sha256"] = sha256_file(path)
    try:
        header = read_png_header(path)
        row.update(width=header["width"], height=header["height"], bit_depth=header["bit_depth"], color_type=header["color_type"])
    except PNGFormatError as exc:
        row.update(width=None, height=None, bit_depth=None, color_type=None, header_error=str(exc))
    return row


def build_png_manifest(index: ImageIndex, workers: int = 8, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per PNG file: id, relative path, size, SHA-256 (streamed), dimensions, bit depth."""
    prev_map = None
    if previous is not None and not previous.empty:
        prev_map = previous.set_index("relative_path").to_dict(orient="index")
    items = [(stem, p) for stem, paths in sorted(index.stem_to_paths.items()) for p in paths]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda it: _manifest_row(it[0], it[1], index.images_dir, prev_map), items))
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values("relative_path").reset_index(drop=True)
    logger.info("Built PNG manifest with %d rows", len(df))
    return df


def manifest_checksum(manifest: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Order-independent checksum of a manifest DataFrame (volatile mtime is excluded)."""
    cols = columns or [c for c in manifest.columns if c != "mtime_ns"]
    canonical = manifest[cols].astype(str).sort_values(cols).to_csv(index=False)
    return sha256_text(canonical)


def verify_png_manifest(saved: pd.DataFrame, index: ImageIndex, workers: int = 8) -> dict[str, Any]:
    """Recompute checksums and compare with a saved manifest (changed, missing and added files)."""
    current = build_png_manifest(index, workers=workers, previous=None)
    s = saved.set_index("relative_path")
    c = current.set_index("relative_path")
    missing = sorted(set(s.index) - set(c.index))
    added = sorted(set(c.index) - set(s.index))
    common = sorted(set(s.index) & set(c.index))
    changed = [r for r in common if s.loc[r, "sha256"] != c.loc[r, "sha256"] or s.loc[r, "file_size"] != c.loc[r, "file_size"]]
    report = {
        "n_saved": len(s),
        "n_current": len(c),
        "missing": missing,
        "added": added,
        "changed": changed,
        "identical": not (missing or added or changed),
    }
    if not report["identical"]:
        logger.warning(
            "PNG manifest verification found differences: %d missing, %d added, %d changed",
            len(missing),
            len(added),
            len(changed),
        )
    return report


def checksum_files(files: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(Path(p)) for name, p in files.items() if Path(p).exists()}


def compute_dataset_id(csv_checksums: Mapping[str, str], png_manifest_checksum: str) -> str:
    """Stable identifier derived from the source CSV checksums and the PNG manifest checksum."""
    payload = "|".join(f"{k}={csv_checksums[k]}" for k in sorted(csv_checksums)) + f"|png={png_manifest_checksum}"
    return "ds-" + sha256_text(payload)[:16]


def write_dataset_manifest(
    path: Path,
    release_id: str,
    csv_checksums: Mapping[str, str],
    png_manifest_checksum: str,
    n_indexed_pngs: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or update ``dataset_manifest.json``; sample/split fields are added in later calls."""
    path = Path(path)
    record = load_json(path) if path.exists() else {"created_utc": datetime.now(timezone.utc).isoformat()}
    record.update(
        dataset_release_id=release_id,
        csv_checksums=dict(csv_checksums),
        png_manifest_checksum=png_manifest_checksum,
        n_indexed_pngs=n_indexed_pngs,
        dataset_id=compute_dataset_id(csv_checksums, png_manifest_checksum),
        checksum_algorithm="sha256",
    )
    if extra:
        record.update(extra)
    snapshot_fields = {k: record.get(k) for k in ("dataset_id", "sample_manifest_checksum", "split_manifest_checksum")}
    record["dataset_manifest_hash"] = sha256_text(str(sorted(snapshot_fields.items())))[:16]
    save_json(record, path)
    return record


class DVCManager:
    """Thin wrapper around the DVC CLI. Only small files are ever tracked."""

    def __init__(self, repo_root: Path, cfg: Mapping[str, Any], remote_path: Path):
        self.repo_root = Path(repo_root)
        self.cfg = cfg["mlops"]["dvc"]
        self.remote_path = Path(remote_path)
        if self.cfg.get("track_full_images", False):
            raise ValueError("track_full_images must be false in Stage 1.")

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["dvc", *args], cwd=self.repo_root, capture_output=True, text=True, check=check)

    def ensure_initialized(self) -> None:
        if not (self.repo_root / ".git").exists():
            raise RuntimeError(
                f"{self.repo_root} is not a Git repository. DVC requires Git: clone the repository or run `git init` first."
            )
        if not (self.repo_root / ".dvc").exists():
            self._run("init")
        remotes = self._run("remote", "list", "--local", check=False).stdout
        if self.cfg["remote_name"] not in remotes:
            self.remote_path.mkdir(parents=True, exist_ok=True)
            # --local keeps the machine-specific remote path out of the public repository.
            self._run("remote", "add", "--local", "-d", self.cfg["remote_name"], str(self.remote_path))

    def stage_files(self, files: Mapping[str, Path]) -> list[Path]:
        """Copy small files into the repository tracking directory and ``dvc add`` each of them."""
        tracked_dir = self.repo_root / self.cfg["tracked_dir"]
        tracked_dir.mkdir(parents=True, exist_ok=True)
        staged = []
        for name, src in files.items():
            src = Path(src)
            if src.suffix.lower() == ".png" or src.is_dir():
                raise ValueError(f"Refusing to DVC-track images or directories: {src}")
            dst = tracked_dir / name
            shutil.copy2(src, dst)
            self._run("add", str(dst.relative_to(self.repo_root)))
            staged.append(dst)
        return staged

    def push(self) -> str:
        return self._run("push").stdout

    def status_cloud(self) -> str:
        return self._run("status", "--cloud", check=False).stdout.strip()

    def remote_in_sync(self) -> bool:
        """True if every tracked file is present in the remote (message wording differs between DVC versions)."""
        text = self.status_cloud().lower()
        return "in sync" in text or "up to date" in text

    def verify_roundtrip(self, staged: Path) -> bool:
        """Delete one tracked workspace copy and restore it via ``dvc pull``; compare checksums.

        Note: ``dvc pull`` may restore from the local cache; ``status_cloud`` confirms remote contents.
        """
        before = sha256_file(staged)
        staged.unlink()
        self._run("pull", str(staged.relative_to(self.repo_root)) + ".dvc")
        return staged.exists() and sha256_file(staged) == before
