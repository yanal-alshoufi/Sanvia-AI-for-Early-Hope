"""YAML configuration loading, validation and hashing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from sanvia.utils.io import save_json, sha256_text, to_jsonable

CONFIG_FILES = ("base.yaml", "data.yaml", "training.yaml", "mlops.yaml", "backbones.yaml")
REQUIRED_BACKBONE_KEYS = {"name", "builder", "weights_enum", "weights_name", "input_size", "feature_dim"}


def deep_update(base: dict, override: Mapping) -> dict:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(config_dir: Path | str, overrides: Mapping | None = None) -> dict[str, Any]:
    """Load and merge the five YAML files, apply overrides, validate and return the config."""
    config_dir = Path(config_dir)
    cfg: dict[str, Any] = {}
    for name in CONFIG_FILES:
        path = config_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing configuration file: {path}")
        with open(path, encoding="utf-8") as fh:
            content = yaml.safe_load(fh) or {}
        overlap = set(cfg) & set(content)
        if overlap:
            raise ValueError(f"Configuration key collision in {name}: {sorted(overlap)}")
        cfg.update(content)
    if overrides:
        deep_update(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if the configuration is inconsistent with the Stage 1 rules."""
    errors: list[str] = []
    if cfg["mlops"]["dvc"].get("track_full_images", False):
        errors.append("mlops.dvc.track_full_images must remain false in Stage 1.")
    if cfg["data"].get("evaluate_official_test", False):
        errors.append("data.evaluate_official_test must be false in Stage 1.")
    seeds = {cfg["project"]["seed"], cfg["training"]["seed"], cfg["data"]["split"]["seed"]}
    if len(seeds) != 1:
        errors.append(f"Seeds are inconsistent across project/training/split: {sorted(seeds)}")
    t = cfg["training"]
    if t["batch_size"] <= 0 or t["effective_batch_size"] % t["batch_size"] != 0:
        errors.append("training.effective_batch_size must be a positive multiple of training.batch_size.")
    if not 0 < cfg["data"]["split"]["val_fraction"] < 1:
        errors.append("data.split.val_fraction must be in (0, 1).")
    if cfg["data"]["age"]["mode"] not in {"auto", "enabled", "disabled"}:
        errors.append("data.age.mode must be auto, enabled or disabled.")
    if cfg["data"]["normalization"]["strategy"] not in {"dtype_range", "global_window"}:
        errors.append("data.normalization.strategy must be dtype_range or global_window.")
    if cfg["data"]["augmentation"].get("vertical_flip"):
        errors.append("Vertical flip is not allowed for mammography augmentation.")
    birads_values = sorted(set(cfg["data"]["labels"]["birads_map"].values()))
    density_values = sorted(set(cfg["data"]["labels"]["density_map"].values()))
    if birads_values != list(range(5)) or density_values != list(range(4)):
        errors.append("Label maps must encode BI-RADS as 0..4 and density as 0..3.")
    names = [b["name"] for b in cfg["backbones"]]
    if len(names) != len(set(names)) or len(names) != 6:
        errors.append(f"Expected six unique backbones, found {names}.")
    for b in cfg["backbones"]:
        missing = REQUIRED_BACKBONE_KEYS - set(b)
        if missing:
            errors.append(f"Backbone {b.get('name')} is missing keys {sorted(missing)}.")
    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))


def config_hash(cfg: Mapping[str, Any], length: int = 12) -> str:
    """Stable hash of a (sub)configuration."""
    text = json.dumps(to_jsonable(cfg), sort_keys=True)
    return sha256_text(text)[:length]


def run_config_subset(cfg: Mapping[str, Any], backbone_name: str) -> dict[str, Any]:
    """Configuration parts that define one experiment; used for the resume-compatibility hash."""
    spec = next(b for b in cfg["backbones"] if b["name"] == backbone_name)
    data = cfg["data"]
    return {
        "backbone": spec,
        "training": cfg["training"],
        "labels": data["labels"],
        "normalization": data["normalization"],
        "preprocessing": data["preprocessing"],
        "augmentation": data["augmentation"],
        "age": data["age"],
        "split": data["split"],
    }


def save_resolved_config(cfg: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_jsonable(cfg), fh, sort_keys=True, allow_unicode=True)
    return path


def save_resolved_config_json(cfg: Mapping[str, Any], path: Path) -> Path:
    return save_json(cfg, path)
