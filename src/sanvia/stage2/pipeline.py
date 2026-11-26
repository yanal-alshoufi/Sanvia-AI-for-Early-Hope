"""One local, content-identified benchmark shared by the CLI and notebook."""

from __future__ import annotations

import gc
import logging
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from filelock import FileLock

from sanvia.mlops.git_metadata import get_git_metadata
from sanvia.models.backbone_factory import get_backbone_spec, weights_metadata
from sanvia.stage2.adapters import ADAPTERS
from sanvia.stage2.audit import audit_sources
from sanvia.stage2.comparison import compare, handoff
from sanvia.stage2.preprocessing import pixels, preprocess
from sanvia.stage2.source_training import train_source
from sanvia.stage2.stage1_artifacts import read_snapshot, require
from sanvia.stage2.target import train_target
from sanvia.stage2.tracking import register_handoff, stage_dvc, tracked
from sanvia.utils.config import config_hash, deep_update
from sanvia.utils.io import load_json, save_json, sha256_file

PRIMARY = ("CBIS-DDSM", "CMMD", "INbreast")
LOG = logging.getLogger(__name__)


def configuration(repo: Path, overrides: dict | None = None) -> dict:
    base = yaml.safe_load((repo / "configs/stage2/default.yaml").read_text())
    cfg = deep_update(base, overrides or {})
    require(
        cfg["mode"] in {"smoke", "pilot", "full"} and (cfg["mode"] == "smoke" or cfg["pretrained"] is True),
        "Real experiments require ImageNet initialization",
    )
    names = cfg["enabled_sources"]
    require(bool(names) and len(set(names)) == len(names) and set(names) <= set(ADAPTERS), "Invalid enabled sources")
    require(
        set(PRIMARY) <= set(names) or (cfg["reduced_sources"] and cfg["name"] != "primary"),
        "Missing primary sources require a separately named reduced-source experiment",
    )
    require(
        cfg["strategies"]
        and set(cfg["strategies"]) <= {"multi-dataset", "sequential"}
        and len(set(cfg["strategies"])) == len(cfg["strategies"]),
        "Invalid strategies",
    )
    require(
        cfg["seeds"] and len(set(cfg["seeds"])) == len(cfg["seeds"]) and all(type(n) is int and n >= 0 for n in cfg["seeds"]),
        "Invalid seeds",
    )
    require(not cfg["pair_override"] or cfg["name"] != "primary", "Backbone overrides require a named controlled experiment")
    require(not cfg.get("synthetic") or cfg["mode"] == "smoke", "Synthetic fixtures require smoke mode")
    require(0 < cfg["validation_fraction"] < 1, "Invalid source validation fraction")
    t = cfg["source_training"]
    require(
        all(isinstance(t[k], int) and t[k] > 0 for k in ("batch_size", "accumulation", "eval_every", "train_updates"))
        and t["workers"] >= 0
        and t["l2sp"] >= 0
        and 0 <= t["replay_fraction"] <= 1,
        "Invalid source training budget",
    )
    require(not t["replay_fraction"] or cfg["strategies"] == ["sequential"], "Replay requires a separate sequential experiment")
    require(
        isinstance(cfg["name"], str) and cfg["name"] and Path(cfg["name"]).name == cfg["name"] and cfg["name"] not in {".", ".."},
        "Invalid experiment name",
    )
    require(
        cfg["selection"]["min_gain"] >= 0
        and all(
            cfg["selection"][k] >= 0
            for k in (
                "max_task_regression",
                "max_qwk_regression",
                "max_minority_recall_regression",
                "max_overfit_gap",
                "bootstrap_iterations",
            )
        ),
        "Invalid preregistered selection thresholds",
    )
    require(cfg["probe_epochs"] > 0, "Linear-probe duration must be positive")
    return cfg


class Benchmark:
    def __init__(self, repo: Path, cfg: dict, device=None):
        self.repo = repo.resolve()
        self.cfg = cfg = configuration(self.repo, cfg)
        output = Path(cfg["output"]).resolve()
        self.mode = cfg["mode"]
        self.base = output / self.mode
        self.base.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(output / ".stage2-writer.lock", timeout=0)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() and not cfg.get("synthetic") else "cpu"))
        if cfg.get("synthetic"):
            from sanvia.stage2.synthetic import smoke_inputs

            with self.lock:
                self.snapshot = smoke_inputs(self.base / "synthetic-fixture", self.repo, cfg)
        else:
            self.snapshot = read_snapshot(Path(cfg["stage1_output"]), Path(cfg["vindr_images"]), cfg["pair_override"])
        self.registry = yaml.safe_load((self.repo / "configs/stage2/tasks.yaml").read_text())
        self.git = get_git_metadata(self.repo)
        self.git.pop("remote", None)
        self.code = config_hash(
            {str(p.relative_to(self.repo)): sha256_file(p) for p in sorted((self.repo / "src/sanvia").rglob("*.py"))}, 64
        )
        self.environment = dict(
            python=platform.python_version(),
            packages={
                n: version(n)
                for n in (
                    "sanvia",
                    "torch",
                    "torchvision",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "pydicom",
                    "Pillow",
                    "mlflow",
                    "dvc",
                )
            },
            device=str(self.device),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        )
        self.key = config_hash(
            dict(
                config=cfg,
                registry=self.registry,
                code=self.code,
                stage1=self.snapshot.report,
                version=self.environment,
                git=self.git["commit"],
                dataset=self.snapshot.data.dataset_manifest_hash,
                target=self.snapshot.config,
            ),
            24,
        )
        self.out = self.base / cfg["name"] / self.key
        self.out.mkdir(parents=True, exist_ok=True)
        if cfg["cache"]:
            Path(cfg["cache"]).mkdir(parents=True, exist_ok=True)
        self.storage = {
            str(path): shutil.disk_usage(path).free for path in [self.out, *([Path(cfg["cache"])] if cfg["cache"] else [])]
        }
        for path, free in self.storage.items():
            require(
                free >= cfg.get("minimum_free_gib", 0) * 2**30,
                f"Insufficient free space in {path}: {free / 2**30:.2f} GiB available",
            )

        self.audit_dir = self.out / "audit"
        self.data = None

    def provenance(self):
        return dict(
            config=self.cfg,
            stage1_baselines=self.snapshot.baselines,
            selected=self.snapshot.selected,
            git=self.git,
            code=self.code,
            environment=self.environment,
            free_bytes_at_start=self.storage,
            stage1_commit=self.snapshot.report["git"].get("commit"),
            target_dataset=self.snapshot.data.dataset_id,
            target_split=self.snapshot.data.split_id,
            target_manifest=self.snapshot.data.dataset_manifest_hash,
            experiment=self.key,
            mode=self.mode,
            sources={
                s: dict(version=d["version"], fingerprint=d["fingerprint"], split_id=d["split_id"])
                for s, d in (self.data or {}).items()
            },
            official_test_used=False,
        )

    def _prepare(self):
        if not self.cfg.get("synthetic") and "CBIS-DDSM" in self.cfg["enabled_sources"]:
            entries = self.cfg["sources"]["CBIS-DDSM"]["annotations"]
            require(
                {entry.get("official_split") for entry in entries} == {"train", "test"},
                "Configure the official CBIS training and test case descriptions before auditing",
            )
        marker = self.audit_dir / "accepted.json"
        if marker.exists():
            accepted = load_json(marker)
            data = {}
            for s in self.cfg["enabled_sources"]:
                folder = self.audit_dir / s
                require(
                    not load_json(self.audit_dir / "critical_issues.json") and not load_json(folder / "critical_issues.json"),
                    "Critical audit issues are unresolved",
                )
                record = load_json(folder / "dataset_manifest.json")
                fingerprint = record.pop("fingerprint")
                require(
                    config_hash(record, 64) == fingerprint == accepted[s],
                    "Prepared manifest changed. Restore the audited snapshot.",
                )
                record["fingerprint"] = fingerprint
                adapter = ADAPTERS[s](self.cfg["sources"][s])
                require(
                    record["annotation_hashes"] == [sha256_file(adapter.annotation_path(e)) for e in adapter.annotations],
                    "Source annotations changed. Configure a new source version and audit.",
                )
                for entry in record["images"].values():
                    require(
                        sha256_file(adapter.root / entry["image_path"]) == entry["sha256"],
                        f"Source image changed since audit: {entry['image_path']}",
                    )
                data[s] = record
            self.data = data
        else:
            self.data = audit_sources(self.cfg, self.registry, self.audit_dir, self.snapshot.png_hashes)
            save_json({s: d["fingerprint"] for s, d in self.data.items()}, marker)
        save_json(self.provenance(), self.out / "reproducibility_report.json")
        if self.cfg["dvc"] and not (self.out / "dvc_status.json").exists():
            stage_dvc(self.repo, self.out, {"stage2_manifest_hashes.json": marker})
        if self.cfg.get("preview_images"):
            for name, d in self.data.items():
                dest = self.audit_dir / name / "preprocessing.png"
                if dest.is_file():
                    continue
                s = next(s for s in d["samples"] if s["split"] == "train" and s["eligible"])
                info = d["images"][next(v for v in s["views"] if v)]
                path = Path(self.cfg["sources"][name]["root"]) / info["image_path"]
                size = get_backbone_spec(self.snapshot.config, self.snapshot.selected[0]).input_size
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(pixels(path)[0], cmap="gray")
                axes[0].set_title("Decoded projection")
                axes[1].imshow(
                    preprocess(path, info["sha256"], size, self.snapshot.config["data"]["preprocessing"])[0], cmap="gray"
                )
                axes[1].set_title("Letterboxed")
                for ax in axes:
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(dest)
                plt.close(fig)
        return self.data

    def plan(self):
        return [
            dict(
                backbone=b,
                seed=k,
                strategy=s,
                mode=self.mode,
                source_order=self.cfg["enabled_sources"],
                training=self.settings(b),
            )
            for b in self.snapshot.selected
            for k in self.cfg["seeds"]
            for s in ["imagenet", *self.cfg["strategies"]]
        ]

    def settings(self, name):
        settings = dict(self.cfg["source_training"])
        settings["batch_size"] = settings["batch_by_backbone"].get(name, settings["batch_size"])
        require(settings["batch_size"] > 0, "Invalid backbone batch size")
        return settings

    def identity(self, b, seed, strategy):
        spec = get_backbone_spec(self.snapshot.config, b)
        return dict(
            benchmark=self.key,
            mode=self.mode,
            backbone=b,
            seed=seed,
            strategy=strategy,
            code=self.code,
            commit=self.git["commit"],
            source_order=self.cfg["enabled_sources"],
            sources={s: d["fingerprint"] for s, d in self.data.items()},
            target_split=self.snapshot.data.split_id,
            weights_identifier=weights_metadata(spec)["weights_identifier"] if self.cfg["pretrained"] else "smoke_random",
        )

    def _experiments(self, action, resume):
        rows = []
        for row in self.plan():
            name, seed, arm = row["backbone"], row["seed"], row["strategy"]
            spec = get_backbone_spec(self.snapshot.config, name)
            out = self.out / name / str(seed) / arm
            identity = self.identity(name, seed, arm)
            src = out / "source/source_adapted_backbone.pt"
            do_source = arm != "imagenet" and action in ("run-all", "train-" + arm)
            if do_source:
                completed = src.parent / "completed.json"
                if completed.exists():
                    meta = load_json(completed)
                    require(
                        meta["identity"] == identity and meta["backbone_sha256"] == sha256_file(src),
                        "Completed source export mismatch",
                    )
                else:
                    LOG.info("Source training %s/%s/%s", name, seed, arm)
                    with tracked(self.out, src.parent, identity, self.cfg["mlflow"]) as tracker:
                        train_source(
                            spec,
                            self.snapshot,
                            self.data,
                            self.cfg,
                            self.settings(name),
                            arm,
                            seed,
                            src.parent,
                            identity,
                            resume,
                            self.device,
                            tracker,
                        )
            if action.startswith("train-"):
                continue
            probe = action == "linear-probe"
            folder = out / ("linear_probe" if probe else "target")
            summary = folder / "target_summary.json"
            if action in {"run-all", "finetune-vindr", "linear-probe"} and not summary.exists():
                if arm != "imagenet":
                    completion = load_json(src.parent / "completed.json")
                    require(
                        completion["identity"] == identity and completion["backbone_sha256"] == sha256_file(src),
                        "Source initialization identity mismatch",
                    )
                LOG.info("Controlled target training %s/%s/%s", name, seed, arm)
                with tracked(self.out, folder, identity, self.cfg["mlflow"]) as tracker:
                    train_target(
                        spec,
                        self.snapshot,
                        seed,
                        self.cfg["pretrained"],
                        src if arm != "imagenet" else None,
                        folder,
                        identity,
                        resume,
                        self.device,
                        tracker,
                        probe_epochs=self.cfg["probe_epochs"] if probe else 0,
                    )
            result = load_json(summary)
            require(
                result["identity"] == identity and sha256_file(Path(result["initial"])) == result["initial_sha256"],
                "Completed target identity mismatch",
            )
            if arm != "imagenet" and not probe:
                save_json(
                    dict(
                        source_export_sha256=result["initial_sha256"],
                        target_protocol=result["protocol"],
                        target_validation=result["validation"],
                        target_directory="../target",
                    ),
                    src.parent / "target_evaluation.json",
                )
            rows.append(result)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return rows

    def execute(self, action: str, resume=True):
        with self.lock:
            self._prepare()
            if action in {"audit", "prepare"}:
                return self.audit_dir
            require(
                action
                in {
                    "run-all",
                    "finetune-vindr",
                    "linear-probe",
                    "evaluate",
                    "compare",
                    "export",
                    "train-sequential",
                    "train-multi-dataset",
                },
                "Unknown benchmark action",
            )
            if action.startswith("train-"):
                require(action[6:] in self.cfg["strategies"], "Requested strategy is disabled")
            if self.mode in {"full", "pilot"} and (
                action.startswith("train-") or action in {"finetune-vindr", "run-all", "linear-probe"}
            ):
                from sanvia.stage2.preflight import preflight

                preflight(self.repo, self.out, self.code)
            rows = self._experiments(action, resume)
            if action in {"evaluate", "compare", "run-all", "export"}:
                if set(self.cfg["strategies"]) != {"sequential", "multi-dataset"}:
                    require(action not in {"compare", "export"}, "A comparison requires both adaptation strategies")
                    return self.out
                result = compare(
                    rows, self.snapshot.selected, self.cfg["seeds"], self.cfg["selection"], self.out / "comparison", self.mode
                )
                if action == "export" or (action == "run-all" and self.mode == "full"):
                    dest = self.out / "stage3_handoff"
                    handoff(rows, result, self.provenance(), dest)
                    if self.cfg["mlflow"]:
                        try:
                            register_handoff(dest, self.out)
                        except Exception as exc:
                            save_json({"error": str(exc)}, dest / "registry_failure.json")
                            raise RuntimeError(f"Handoff saved; local model registration failed: {exc}") from exc
            return self.out
