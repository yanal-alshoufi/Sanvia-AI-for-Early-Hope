"""Data audit: CSV schema, PNG integrity and intensity statistics, annotations and findings."""

from __future__ import annotations

import ast
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402

from sanvia.data.labels import LabelSpace, encode_series  # noqa: E402
from sanvia.data.loading import normalize_id  # noqa: E402
from sanvia.data.png_io import PNGFormatError, load_png_array  # noqa: E402
from sanvia.utils.io import save_json  # noqa: E402

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- images
def _audit_one_image(
    rel: str, images_dir: Path, header: Mapping[str, Any], do_stats: bool, percentiles: list[float], stride: int
) -> dict[str, Any]:
    path = images_dir / rel
    row: dict[str, Any] = {
        "relative_path": rel,
        "readable": True,
        "error": None,
        "stats_computed": False,
        "width": header.get("width"),
        "height": header.get("height"),
        "bit_depth": header.get("bit_depth"),
        "color_type": header.get("color_type"),
    }
    if header.get("header_error"):
        row.update(readable=False, error=header["header_error"])
        return row
    try:
        with Image.open(path) as im:
            im.verify()  # checks chunk structure and CRCs for every file
    except Exception as exc:  # noqa: BLE001
        row.update(readable=False, error=f"verify failed: {exc}")
        return row
    if not do_stats:
        return row
    try:
        array, info = load_png_array(path)
    except (PNGFormatError, OSError, ValueError) as exc:
        row.update(readable=False, error=str(exc))
        return row
    flat = array.ravel()[::stride].astype(np.float64)
    h, w = array.shape
    corners = np.concatenate(
        [
            array[: h // 10, : w // 10].ravel(),
            array[: h // 10, -(w // 10) :].ravel(),
            array[-(h // 10) :, : w // 10].ravel(),
            array[-(h // 10) :, -(w // 10) :].ravel(),
        ]
    )
    center = array[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    row.update(
        stats_computed=True,
        pil_mode=info["pil_mode"],
        dtype=info["dtype"],
        channels=info["channels"],
        channels_identical=info["channels_identical"],
        min=float(array.min()),
        max=float(array.max()),
        mean=float(flat.mean()),
        std=float(flat.std()),
        zero_fraction=float((flat == 0).mean()),
        is_empty=bool(array.max() == array.min()),
        corner_mean=float(corners.mean()),
        center_mean=float(center.mean()),
    )
    for p, v in zip(percentiles, np.percentile(flat, percentiles)):
        row[f"p{p}"] = float(v)
    return row


def audit_images(
    png_manifest: pd.DataFrame,
    images_dir: Path,
    sample_size: int | None,
    seed: int,
    workers: int,
    percentiles: list[float],
    stride: int,
) -> pd.DataFrame:
    """Integrity check for every PNG plus pixel statistics for all (or a seeded sample of) images."""
    rng = np.random.RandomState(seed)
    n = len(png_manifest)
    stats_idx = set(range(n)) if sample_size is None or sample_size >= n else set(rng.choice(n, sample_size, replace=False))
    jobs = [(i, r) for i, r in enumerate(png_manifest.to_dict(orient="records"))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(
            pool.map(
                lambda j: _audit_one_image(j[1]["relative_path"], images_dir, j[1], j[0] in stats_idx, percentiles, stride), jobs
            )
        )
    audit = pd.DataFrame(rows)
    audit.insert(0, "image_id", png_manifest["image_id"].values)
    return audit


def summarize_image_audit(audit: pd.DataFrame, expected_total: int = 20000) -> dict[str, Any]:
    stats = audit[audit["stats_computed"]] if "stats_computed" in audit else audit.iloc[0:0]
    summary: dict[str, Any] = {
        "n_images": len(audit),
        "expected_images_note": f"Expected approx. {expected_total}; actual count reported.",
        "n_unreadable": int((~audit["readable"]).sum()),
        "bit_depth_counts": audit["bit_depth"].value_counts(dropna=False).to_dict(),
        "size_counts_top10": audit.assign(size=audit["width"].astype(str) + "x" + audit["height"].astype(str))["size"]
        .value_counts()
        .head(10)
        .to_dict(),
        "n_stats_computed": int(len(stats)),
    }
    if len(stats):
        summary.update(
            dtype_counts=stats["dtype"].value_counts().to_dict(),
            global_min=float(stats["min"].min()),
            global_max=float(stats["max"].max()),
            max_by_bit_depth={str(k): float(v) for k, v in stats.groupby("bit_depth")["max"].max().items()},
            n_empty=int(stats["is_empty"].sum()),
            n_non_identical_channels=int((~stats["channels_identical"].astype(bool)).sum()),
            n_possible_inverted=int((stats["corner_mean"] > stats["center_mean"]).sum()),
        )
    return summary


# ------------------------------------------------------------------------ annotations
def audit_annotations(
    breast: pd.DataFrame,
    cols: Mapping[str, str],
    birads_space: LabelSpace,
    density_space: LabelSpace,
    laterality_values: list[str],
    view_values: list[str],
) -> dict[str, Any]:
    """Breast-level annotation audit. Returns per-breast audit, distributions, duplicates and a summary."""
    sid, lat, view, img = cols["study_id"], cols["laterality"], cols["view_position"], cols["image_id"]
    df = breast.copy()
    df["birads_encoded"], df["birads_status"] = encode_series(df[cols["birads"]], birads_space)
    df["density_encoded"], df["density_status"] = encode_series(df[cols["density"]], density_space)
    df["breast_key"] = df[sid].astype(str) + "_" + df[lat].astype(str)

    dup_view = df.duplicated([sid, lat, view], keep=False)
    duplicates = df[dup_view | df.duplicated(keep=False)].copy()
    duplicates["duplicate_type"] = np.where(df.duplicated(keep=False)[duplicates.index], "exact_row", "same_study_lat_view")

    per_breast = (
        df.groupby([sid, lat])
        .agg(
            n_records=(img, "size"),
            n_cc=(view, lambda s: int((s == view_values[0]).sum())),
            n_mlo=(view, lambda s: int((s == view_values[1]).sum())),
            n_distinct_birads=(cols["birads"], lambda s: s.dropna().nunique()),
            n_distinct_density=(cols["density"], lambda s: s.dropna().nunique()),
            birads_values=(cols["birads"], lambda s: "|".join(sorted(map(str, s.dropna().unique())))),
            density_values=(cols["density"], lambda s: "|".join(sorted(map(str, s.dropna().unique())))),
            n_birads_missing=("birads_status", lambda s: int((s == "missing").sum())),
            n_density_missing=("density_status", lambda s: int((s == "missing").sum())),
        )
        .reset_index()
    )
    per_breast["birads_inconsistent"] = (per_breast["n_distinct_birads"] > 1) | (
        (per_breast["n_birads_missing"] > 0) & (per_breast["n_birads_missing"] < per_breast["n_records"])
    )
    per_breast["density_inconsistent"] = (per_breast["n_distinct_density"] > 1) | (
        (per_breast["n_density_missing"] > 0) & (per_breast["n_density_missing"] < per_breast["n_records"])
    )

    dist_rows = []
    for task, col, status_col in (("birads", cols["birads"], "birads_status"), ("density", cols["density"], "density_status")):
        for split_name, part in [("all", df), *df.groupby(cols["split"])]:
            for label, g in part.groupby(part[col].fillna("<MISSING>").astype(str)):
                dist_rows.append(
                    {
                        "task": task,
                        "official_split": split_name,
                        "label": label,
                        "status": g[status_col].iloc[0],
                        "n_records": len(g),
                        "n_breasts": g["breast_key"].nunique(),
                    }
                )
    distribution = pd.DataFrame(dist_rows)

    unknown = {
        t: sorted(df.loc[df[f"{t}_status"] == "unknown", c].astype(str).unique().tolist())
        for t, c in (("birads", cols["birads"]), ("density", cols["density"]))
    }
    per_study = df.groupby(sid).size()
    summary = {
        "n_records": len(df),
        "n_unique_studies": int(df[sid].nunique()),
        "n_unique_images": int(df[img].nunique()),
        "n_unique_breasts": int(df["breast_key"].nunique()),
        "records_per_study": per_study.describe().to_dict(),
        "records_per_breast": per_breast["n_records"].describe().to_dict(),
        "view_counts": df[view].value_counts(dropna=False).to_dict(),
        "laterality_counts": df[lat].value_counts(dropna=False).to_dict(),
        "unexpected_laterality_values": sorted(set(df[lat].dropna().astype(str)) - set(laterality_values)),
        "unexpected_view_values": sorted(set(df[view].dropna().astype(str)) - set(view_values)),
        "studies_without_four_images": int((per_study != 4).sum()),
        "birads_missing": int((df["birads_status"] == "missing").sum()),
        "density_missing": int((df["density_status"] == "missing").sum()),
        "unknown_labels": unknown,
        "n_unknown_label_records": int((df["birads_status"] == "unknown").sum() + (df["density_status"] == "unknown").sum()),
        "n_duplicate_records": int(len(duplicates)),
        "n_breasts_birads_inconsistent": int(per_breast["birads_inconsistent"].sum()),
        "n_breasts_density_inconsistent": int(per_breast["density_inconsistent"].sum()),
        "official_split_counts_studies": df.groupby(cols["split"])[sid].nunique().to_dict(),
    }
    return {"per_breast": per_breast, "distribution": distribution, "duplicates": duplicates, "summary": summary}


# --------------------------------------------------------------------------- findings
def _parse_categories(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [str(v) for v in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [str(value)]


def audit_findings(finding: pd.DataFrame, png_manifest: pd.DataFrame, image_col: str = "image_id") -> dict[str, Any]:
    """Validate finding categories and bounding boxes. Audit only; never used for training in Stage 1."""
    summary: dict[str, Any] = {"n_findings": len(finding), "columns": list(finding.columns)}
    cat_col = next((c for c in ("finding_categories", "finding_category") if c in finding.columns), None)
    if cat_col:
        cats = finding[cat_col].map(_parse_categories)
        summary["category_counts"] = pd.Series([c for lst in cats for c in lst]).value_counts().to_dict()
    box_cols = ["xmin", "ymin", "xmax", "ymax"]
    if not set(box_cols).issubset(finding.columns):
        summary["bbox_note"] = f"Bounding box columns {box_cols} not all present."
        return summary
    boxes = finding[box_cols].apply(pd.to_numeric, errors="coerce")
    summary["bbox_format"] = "absolute pixel coordinates (xmin, ymin, xmax, ymax) assumed; verified by ordering checks"
    summary["n_malformed_coordinates"] = int(boxes.isna().any(axis=1).sum())
    ordered = (boxes["xmin"] < boxes["xmax"]) & (boxes["ymin"] < boxes["ymax"])
    summary["n_invalid_ordering"] = int((~ordered & ~boxes.isna().any(axis=1)).sum())
    summary["n_negative_coordinates"] = int((boxes < 0).any(axis=1).sum())
    dims = (
        png_manifest.assign(image_id=png_manifest["image_id"].astype(str))
        .drop_duplicates("image_id")
        .set_index("image_id")[["width", "height"]]
    )
    ids = finding[image_col].map(normalize_id)
    png_w, png_h = ids.map(dims["width"]), ids.map(dims["height"])
    summary["n_without_png"] = int(png_w.isna().sum())
    summary["n_outside_png_bounds"] = int(((boxes["xmax"] > png_w) | (boxes["ymax"] > png_h)).sum())
    if {"width", "height"}.issubset(finding.columns):
        summary["n_outside_csv_bounds"] = int(((boxes["xmax"] > finding["width"]) | (boxes["ymax"] > finding["height"])).sum())
        summary["csv_dims_differ_from_png"] = int(((finding["width"] != png_w) | (finding["height"] != png_h)).sum())
    return summary


# -------------------------------------------------------------------------------- plots
def plot_class_distributions(distribution: pd.DataFrame, out_dir: Path) -> list[Path]:
    out = []
    for task in ("birads", "density"):
        data = distribution[(distribution["task"] == task) & (distribution["official_split"] != "all")]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=data, x="label", y="n_breasts", hue="official_split", ax=ax)
        ax.set_title(f"{task.upper()} distribution (breasts) by official split")
        ax.tick_params(axis="x", rotation=30)
        path = out_dir / f"class_distribution_{task}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out.append(path)
    return out


def plot_representative_images(png_manifest: pd.DataFrame, images_dir: Path, out_dir: Path, n: int = 6, seed: int = 0) -> Path:
    """Representative images with intensity histograms (computed from the actual decoded values)."""
    pick = png_manifest.sample(min(n, len(png_manifest)), random_state=seed)
    fig, axes = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 6), squeeze=False)
    for j, (_, r) in enumerate(pick.iterrows()):
        try:
            arr, info = load_png_array(images_dir / r["relative_path"])
        except (PNGFormatError, OSError, ValueError) as exc:
            axes[0, j].set_title(f"unreadable: {exc}"[:30])
            continue
        axes[0, j].imshow(arr, cmap="gray")
        axes[0, j].axis("off")
        axes[0, j].set_title(f"{info['dtype']} {arr.shape[1]}x{arr.shape[0]}", fontsize=8)
        axes[1, j].hist(arr.ravel()[::8], bins=100, log=True)
        axes[1, j].set_xlabel("intensity")
    path = out_dir / "representative_images.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_data_audit(
    tables: Mapping[str, pd.DataFrame],
    png_manifest: pd.DataFrame,
    images_dir: Path,
    out_dir: Path,
    schema_report: Mapping[str, Any],
    cfg: Mapping[str, Any],
    match_df: pd.DataFrame,
    unexpected_df: pd.DataFrame,
    birads_space: LabelSpace,
    density_space: LabelSpace,
) -> dict[str, Any]:
    """Run all audits and persist the required reports. Returns a dictionary of results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    acfg, dcfg = cfg["data"]["audit"], cfg["data"]
    cols = dcfg["columns"]
    sc = dcfg["sample_construction"]
    image_audit = audit_images(
        png_manifest,
        images_dir,
        acfg["stats_sample_size"],
        cfg["project"]["seed"],
        acfg["workers"],
        acfg["percentiles"],
        acfg["pixel_stride"],
    )
    image_summary = summarize_image_audit(image_audit)
    ann = audit_annotations(
        tables["breast"], cols, birads_space, density_space, sc["laterality_values"], list(sc["view_values"].values())
    )
    findings = audit_findings(tables["finding"], png_manifest, cols["image_id"])
    corrupted = image_audit[~image_audit["readable"]]
    ambiguous = match_df[match_df["status"] == "ambiguous"]
    missing = match_df[match_df["status"] == "missing"]

    image_audit.to_csv(out_dir / "image_audit.csv", index=False)
    ann["per_breast"].to_csv(out_dir / "annotation_audit.csv", index=False)
    ann["distribution"].to_csv(out_dir / "label_distribution.csv", index=False)
    ann["duplicates"].to_csv(out_dir / "duplicate_records.csv", index=False)
    missing.to_csv(out_dir / "missing_images.csv", index=False)
    corrupted.to_csv(out_dir / "corrupted_images.csv", index=False)
    ambiguous.to_csv(out_dir / "ambiguous_image_mappings.csv", index=False)
    unexpected_df.to_csv(out_dir / "unexpected_image_files.csv", index=False)
    save_json(schema_report, out_dir / "csv_schema_report.json")
    save_json(image_summary, out_dir / "image_audit_summary.json")
    save_json(findings, out_dir / "finding_audit.json")
    summary = {
        "images": image_summary,
        "annotations": ann["summary"],
        "findings": findings,
        "mapping": {
            "n_annotation_image_ids": len(match_df),
            "n_missing": len(missing),
            "n_ambiguous": len(ambiguous),
            "n_unexpected_png_files": len(unexpected_df),
            "n_duplicate_stems": int((match_df["n_candidates"] > 1).sum()),
        },
        "n_corrupted_images": len(corrupted),
    }
    save_json(summary, out_dir / "data_audit_summary.json")
    plots = plot_class_distributions(ann["distribution"], out_dir)
    plots.append(plot_representative_images(png_manifest, images_dir, out_dir))
    return {"summary": summary, "image_audit": image_audit, "annotation_audit": ann, "corrupted": corrupted, "plots": plots}
