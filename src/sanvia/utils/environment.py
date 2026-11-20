"""Environment and hardware information."""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from typing import Any

MAIN_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "pillow",
    "matplotlib",
    "seaborn",
    "pyyaml",
    "mlflow",
    "dvc",
    "pytest",
    "ruff",
]


def package_versions(packages: list[str] | None = None) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in packages or MAIN_PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def gpu_summary() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=props.name,
            gpu_total_memory_gb=round(props.total_memory / 1024**3, 2),
            gpu_count=torch.cuda.device_count(),
            cudnn_version=torch.backends.cudnn.version(),
        )
    else:
        info.update(gpu_name=None, gpu_total_memory_gb=None, gpu_count=0, cudnn_version=None)
    return info


def collect_environment() -> dict[str, Any]:
    """Collect Python, library, OS and GPU information for reproducibility logs."""
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "gpu": gpu_summary(),
    }
