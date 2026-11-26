"""Normalize source annotations, bind labels to their scopes, and gate source training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pydicom.errors import InvalidDicomError
from sklearn.model_selection import train_test_split

from sanvia.stage2.adapters import ADAPTERS
from sanvia.stage2.preprocessing import pixels
from sanvia.stage2.stage1_artifacts import require
from sanvia.utils.config import config_hash
from sanvia.utils.io import save_json, sha256_file


def namespaced(*parts: str) -> str:
    return "/".join(quote(str(p), safe="") for p in parts)


def encode(rows: list[dict], task: dict, policy: str) -> tuple[dict, str]:
    """Conservative positive-only BCE and multiclass masks. Exclusions retain their reason."""
    raw = [r.get(task["field"], "").strip() for r in rows]
    classes = task["classes"]
    mapping = task["mapping"]
    unknown = set(raw) - set(mapping) - set(task.get("exclude", [])) - {""}
    require(not unknown, f"Unknown {task['field']} vocabulary {sorted(unknown)}")
    known = {classes.index(mapping[v]) for v in raw if v in mapping}
    if task["loss"] == "bce":
        y = [int(i in known) for i in range(len(classes))]
        require(
            task.get("positive_only") is True, "BCE requires verified element-level negative semantics or positive-only masks"
        )
        return dict(y=y, mask=[bool(v) for v in y]), "positive-only"
    if (
        task["field"] == "pathology"
        and policy == "any_malignant"
        and "malignant" in classes
        and classes.index("malignant") in known
    ):
        return dict(y=classes.index("malignant"), mask=True), "any annotated malignant lesion"
    if len(known) == 1 and all(v in mapping for v in raw):
        return dict(y=known.pop(), mask=True), ""
    return dict(y=-1, mask=False), "mixed" if len(known) > 1 else "missing or excluded"


def assign_splits(samples: list[dict], seed: int, fraction: float) -> tuple[dict, str]:
    """Split eligible patient groups once, keeping all official test groups out of development."""
    patients = {}
    labels = defaultdict(set)
    eligible = set()
    for s in samples:
        p = s["patient_id"]
        require(patients.setdefault(p, s["official_split"]) == s["official_split"], "Patient crosses official source partitions")
        if s["eligible"] and s["official_split"] != "test":
            eligible.add(p)
            for k, values in s["tasks"].items():
                for v in values:
                    if v["mask"] is True:
                        labels[p].add((k, v["y"]))
    ids = sorted(eligible)
    require(len(ids) >= 2, "Need at least two eligible source patient groups for train and validation")
    signatures = [str(sorted(labels[p])) for p in ids]
    counts = Counter(signatures)
    nval = int(np.ceil(len(ids) * fraction))
    strat = min(counts.values()) >= 2 and len(counts) <= min(nval, len(ids) - nval)
    _, val = train_test_split(ids, random_state=seed, test_size=fraction, stratify=signatures if strat else None)
    final = {
        p: ("official_test" if part == "test" else "excluded" if p not in eligible else "validation" if p in val else "train")
        for p, part in patients.items()
    }
    return final, "joint observed class signature" if strat else "unstratified (sparse patient strata)"


def save_table(out: Path, name: str, rows: list[dict], columns=()) -> None:
    pd.DataFrame(rows, columns=None if rows else columns).to_csv(out / (name + ".csv"), index=False)


def build_source(name: str, cfg: dict, tasks: dict, out: Path, seed: int, fraction: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tasks = dict(sorted(tasks.items()))
    for key, t in tasks.items():
        require(
            t["scope"] in {"breast", "image"}
            and t["loss"] in {"ce", "bce"}
            and t["weight"] > 0
            and len(t["classes"]) == len(set(t["classes"])) >= 2
            and set(t["mapping"].values()) <= set(t["classes"]),
            f"Invalid task {name}/{key}",
        )
    adapter = ADAPTERS[name](cfg)
    rows = adapter.records()
    require(bool(rows), f"No annotations for {name}")
    images = {}
    by_breast = defaultdict(lambda: defaultdict(list))
    issues = []
    excluded = []
    lesions = []
    observed = set()
    duplicates = 0
    patients = {}
    studies = {}
    sample_records = []
    labels = []
    label_counts = Counter()
    role_counts = Counter()
    hashes = defaultdict(list)
    candidates = defaultdict(list)
    warnings = []
    if name == "CBIS-DDSM":
        for role in ("roi", "crop"):
            derivatives = {r[role].strip() for r in rows if r.get(role, "").strip()}
            role_counts[role] = len(derivatives)
            missing = sum(not (adapter.root / rel).is_file() for rel in derivatives)
            if missing:
                warnings.append(
                    f"{missing} {role} references unavailable at their annotated paths; these derivatives are excluded from primary training."
                )
    all_sources_disjoint = "Independent de-identification prevents verification of cross-dataset patient identity."
    policy = cfg.get("pathology_aggregation", "unanimous")
    require(policy in {"unanimous", "any_malignant"}, "Unknown pathology aggregation policy")
    for r in rows:
        key = config_hash(r, 64)
        if key in observed:
            duplicates += 1
            continue
        observed.add(key)
        role_counts[r["image_type"]] += 1
        if r["image_type"] in {"mask", "crop"}:
            excluded.append(dict(image=r["image"], reason=r["image_type"]))
            continue
        require(
            r["image_type"] == "full" and all(r.get(k) for k in ("patient", "study", "image")),
            "Unresolved full image, patient, image or examination ID",
        )
        require(r["side"] in {"L", "R"} and r["view"] in {"CC", "MLO"}, f"Invalid laterality/view in {r['annotation']}")
        require(r["official_split"] in {"train", "test", "unspecified"}, "Invalid official partition")
        patient = namespaced(name, r["patient"])
        study = namespaced(name, r["patient"], r["study"])
        breast = namespaced(name, r["patient"], r["study"], r["side"])
        image = namespaced(name, r["image"])
        path = Path(r["path"])
        rel = path.resolve().relative_to(adapter.root).as_posix()
        identity = dict(
            dataset_id=name,
            dataset_version=cfg["version"],
            image_id=image,
            raw_image_id=r["image"],
            patient_id=patient,
            study_id=study,
            breast_id=breast,
            image_path=rel,
            view_position=r["view"],
            laterality=r["side"],
            image_type="full",
            official_split=r["official_split"],
        )
        if image in images:
            require(all(images[image][k] == v for k, v in identity.items()), "Conflicting image linkage")
        else:
            pixel, meta = pixels(path)
            if cfg.get("decoded_hashes"):
                meta["decoded_sha256"] = hashlib.sha256(str(pixel.shape).encode() + pixel.astype("<f4").tobytes()).hexdigest()
                candidates[meta["decoded_sha256"]].append(image)
            if cfg.get("near_duplicates"):
                thumbnail = np.asarray(Image.fromarray(pixel).resize((9, 8), Image.Resampling.BILINEAR))
                meta["difference_hash"] = np.packbits(thumbnail[:, 1:] > thumbnail[:, :-1]).tobytes().hex()
                candidates[meta["difference_hash"]].append(image)
            images[image] = dict(identity, **meta, sha256=sha256_file(path))
            hashes[images[image]["sha256"]].append(image)
        require(
            patients.setdefault(
                patient, dict(patient_id=patient, raw_patient_id=r["patient"], official_split=r["official_split"])
            )["official_split"]
            == r["official_split"],
            "Patient crosses official source partitions",
        )
        studies[study] = dict(study_id=study, raw_study_id=r["study"], patient_id=patient)
        by_breast[breast][image].append(r)
        if r.get("lesion"):
            lesions.append(
                dict(
                    image_id=image,
                    lesion_id=namespaced(name, r["study"], r["side"], r["lesion"]),
                    raw_lesion_id=r["lesion"],
                    pathology_raw=r.get("pathology"),
                    finding_raw=r.get("finding"),
                    crop=r.get("crop"),
                    roi=r.get("roi"),
                )
            )
    duplicate_rows = [dict(sha256=h, images=json.dumps(v)) for h, v in hashes.items() if len(v) > 1]
    duplicate_rows.extend(dict(sha256=None, candidate_hash=h, images=json.dumps(v)) for h, v in candidates.items() if len(v) > 1)
    if duplicate_rows:
        issues.append("Duplicate candidates with different image identities require review or documented quarantine")
    for breast, linked in sorted(by_breast.items()):
        ids = list(linked)
        first = images[ids[0]]
        views = [images[i]["view_position"] for i in ids]
        require(
            len(views) == len(set(views)), f"Multiple distinct images in one breast/view: {breast}. Review examination linkage."
        )
        slots = [ids[views.index(v)] if v in views else None for v in ("CC", "MLO")]
        sample = dict(
            sample_id=breast,
            patient_id=first["patient_id"],
            study_id=first["study_id"],
            official_split=first["official_split"],
            views=slots,
            tasks={},
        )
        valid = False
        for task, t in tasks.items():
            groups = [sum(linked.values(), [])] if t["scope"] == "breast" else [linked[i] if i else [] for i in slots]
            values = []
            for slot, group in enumerate(groups):
                result, reason = encode(group, t, policy)
                values.append(result)
                mask = result["mask"]
                has = any(mask) if isinstance(mask, list) else mask
                valid |= has
                if has:
                    if isinstance(mask, list):
                        for j, m in enumerate(mask):
                            if m:
                                label_counts[(task, t["classes"][j])] += 1
                    else:
                        label_counts[(task, t["classes"][result["y"]])] += 1
                labels.append(
                    dict(
                        sample_id=breast,
                        task=task,
                        scope=t["scope"],
                        slot=slot,
                        raw=json.dumps([r.get(t["field"]) for r in group]),
                        label_source=json.dumps(sorted({r["annotation"] for r in group})),
                        valid=has,
                        reason=reason,
                        **result,
                    )
                )
            sample["tasks"][task] = values
        sample["eligible"] = bool(valid)
        if not valid:
            excluded.append(dict(sample_id=breast, reason="no valid supervision"))
        sample_records.append(sample)
    final, method = assign_splits(sample_records, seed, fraction)
    for s in sample_records:
        s["split"] = final[s["patient_id"]]
    missing = [r for r in labels if not r["valid"]]
    counts = [dict(task=k[0], label=k[1], count=v) for k, v in sorted(label_counts.items())]
    save_table(out, "patients", list(patients.values()))
    save_table(out, "studies", list(studies.values()))
    save_table(out, "breasts", [{k: v for k, v in s.items() if k != "tasks"} for s in sample_records])
    save_table(out, "lesions", lesions, ["image_id", "lesion_id"])
    for img in images.values():
        img["final_split"] = final[img["patient_id"]]
        img["acquisition_type"] = {
            "CBIS-DDSM": "digitized film",
            "CMMD": "digital",
            "INbreast": "digital",
            "BCDR-FM": "digitized film",
            "BCDR-DM": "digital",
        }[name]
    save_table(out, "task_labels", labels)
    save_table(out, "original_annotations", rows)
    save_table(out, "split_assignments", [dict(patient_id=p, final_split=f) for p, f in sorted(final.items())])
    save_table(out, "image_manifest", list(images.values()))
    save_table(out, "class_distributions", counts, ["task", "label", "count"])
    save_table(out, "missing_labels", missing, ["sample_id", "task", "reason"])
    save_table(out, "duplicate_candidates", duplicate_rows, ["sha256", "images"])
    save_table(out, "excluded_samples", excluded, ["sample_id", "reason"])
    view_counts = Counter("+".join(view for view, i in zip(("CC", "MLO"), s["views"]) if i) for s in sample_records)
    save_table(out, "view_distributions", [dict(views=k, count=v) for k, v in view_counts.items()])
    for title, values in [
        ("class_distributions", {f"{r['task']}/{r['label']}": r["count"] for r in counts}),
        ("sample_availability", view_counts),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(values), values.values())
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out / (title + ".png"))
        plt.close(fig)
    summary = dict(
        patients=len(patients),
        studies=len(studies),
        breasts=len(sample_records),
        images=len(images),
        lesion_links=len(lesions),
        duplicate_metadata_rows=duplicates,
        multiple_lesion_images=sum(
            len({r.get("lesion") for r in group}) > 1 for links in by_breast.values() for group in links.values()
        ),
        roles=role_counts,
        excluded=len(excluded),
        missing_task_slots=len(missing),
        split_method=method,
        warnings=warnings,
        view_counts=view_counts,
        formats=dict(Counter(i["image_format"] for i in images.values())),
        version=cfg["version"],
        mixed_labels=dict(Counter(r["task"] for r in labels if r["reason"] == "mixed")),
        cross_dataset_patient_identity=all_sources_disjoint,
        access=cfg.get("access_reference", "See docs/stage2.md for release-specific access requirements"),
    )
    save_table(out, "dataset_statistics", [summary])
    save_json(summary, out / "audit_summary.json")
    save_json(issues, out / "critical_issues.json")
    require(not issues, f"{name}: {issues}")
    data = dict(
        dataset=name,
        version=cfg["version"],
        tasks=tasks,
        images=images,
        samples=sample_records,
        split_seed=seed,
        validation_fraction=fraction,
        split_id=config_hash(final, 64),
        annotation_hashes=[sha256_file(adapter.annotation_path(e)) for e in adapter.annotations],
        pathology_aggregation=policy,
    )
    data["fingerprint"] = config_hash(data, 64)
    save_json(data, out / "dataset_manifest.json")
    return data


def audit_sources(cfg: dict, registry: dict, out: Path, target_hashes: set[str]) -> dict:
    """Persist readable failure reports even if a source cannot be decoded or joined."""
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    issues = []
    collisions = []
    owner = dict.fromkeys(target_hashes, "VinDr manifest")
    for name in cfg["enabled_sources"]:
        source = cfg["sources"][name]
        try:
            data = build_source(
                name,
                source,
                source.get("tasks", registry.get(name, {})),
                out / name,
                cfg["split_seed"],
                cfg["validation_fraction"],
            )
            for img in data["images"].values():
                h = img["sha256"]
                if h in owner:
                    collisions.append(dict(dataset=name, other=owner[h], sha256=h))
                owner[h] = name
                for key in ("decoded_sha256", "difference_hash"):
                    if key in img:
                        value = key + ":" + img[key]
                        if value in owner:
                            collisions.append(dict(dataset=name, other=owner[value], candidate_hash=value))
                        owner[value] = name
            result[name] = data
        except (ValueError, KeyError, OSError, RuntimeError, InvalidDicomError, AttributeError) as exc:
            issues.append(f"{name}: {exc}")
            save_json([str(exc)], out / name / "critical_issues.json")
    if collisions:
        issues.append("Exact cross-dataset image duplicates require review")
    save_table(out, "duplicate_candidates", collisions, ["dataset", "other", "sha256"])
    save_json(issues, out / "critical_issues.json")
    require(not issues, "Source integrity gate failed: " + " | ".join(issues))
    return result
