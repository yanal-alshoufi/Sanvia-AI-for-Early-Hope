"""Breast-level sample construction: one sample per (study_id, laterality) with a CC and an MLO image."""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from sanvia.data.labels import LabelSpace, encode_series
from sanvia.data.loading import normalize_id

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "sample_id",
    "study_id",
    "laterality",
    "cc_image_id",
    "mlo_image_id",
    "cc_image_path",
    "mlo_image_path",
    "cc_image_relpath",
    "mlo_image_relpath",
    "cc_series_id",
    "mlo_series_id",
    "birads_original",
    "birads_encoded",
    "birads_valid",
    "density_original",
    "density_encoded",
    "density_valid",
    "age",
    "age_missing_flag",
    "official_split",
    "final_split",
]


def build_breast_samples(
    breast: pd.DataFrame,
    match_df: pd.DataFrame,
    corrupted_ids: set[str],
    cols: Mapping[str, str],
    birads_space: LabelSpace,
    density_space: LabelSpace,
    laterality_values: list[str],
    view_values: Mapping[str, str],
    study_split: Mapping[str, str],
    official_split: Mapping[str, str],
    study_age: pd.DataFrame | None,
    allow_unknown_labels: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(samples, excluded)``. No view is substituted and no conflict is resolved silently."""
    sid, lat, view, img, series = cols["study_id"], cols["laterality"], cols["view_position"], cols["image_id"], cols["series_id"]
    df = breast.copy()
    df[sid] = df[sid].astype(str)
    df["_img"] = df[img].map(normalize_id)
    df["_b"], df["_bs"] = encode_series(df[cols["birads"]], birads_space)
    df["_d"], df["_ds"] = encode_series(df[cols["density"]], density_space)
    matches = match_df.set_index("image_id")
    age_map = {} if study_age is None else study_age.set_index("study_id")["age"].to_dict()
    cc_v, mlo_v = view_values["cc"], view_values["mlo"]

    samples, excluded = [], []
    for (study, lateral), grp in df.groupby([sid, lat], sort=True):
        reasons: list[str] = []
        if lateral not in laterality_values:
            reasons.append(f"unexpected_laterality:{lateral}")
        cc, mlo = grp[grp[view] == cc_v], grp[grp[view] == mlo_v]
        other = set(grp[view].astype(str)) - {cc_v, mlo_v}
        if other:
            reasons.append(f"unexpected_view_position:{sorted(other)}")
        if len(cc) == 0:
            reasons.append("missing_cc")
        if len(mlo) == 0:
            reasons.append("missing_mlo")
        if len(cc) > 1:
            reasons.append(f"duplicate_cc_records:{len(cc)}")
        if len(mlo) > 1:
            reasons.append(f"duplicate_mlo_records:{len(mlo)}")
        if not reasons:
            c, m = cc.iloc[0], mlo.iloc[0]
            for name, row in (("cc", c), ("mlo", m)):
                status = matches["status"].get(row["_img"], "missing") if row["_img"] in matches.index else "missing"
                if status != "ok":
                    reasons.append(f"{name}_image_{status}")
                elif row["_img"] in corrupted_ids:
                    reasons.append(f"{name}_image_unreadable")
            if c["_img"] == m["_img"]:
                reasons.append("cc_and_mlo_share_image_id")
            for task, _orig_col, enc, st in (("birads", cols["birads"], "_b", "_bs"), ("density", cols["density"], "_d", "_ds")):
                if c[st] == "unknown" or m[st] == "unknown":
                    if not allow_unknown_labels:
                        reasons.append(f"{task}_unknown_label")
                elif (c[st] != m[st]) or (c[st] == "ok" and c[enc] != m[enc]):
                    reasons.append(f"{task}_conflict_between_views")
            if not reasons and c["_bs"] != "ok" and c["_ds"] != "ok":
                reasons.append("no_valid_target")
        if reasons:
            excluded.append(
                {
                    "study_id": study,
                    "laterality": lateral,
                    "n_records": len(grp),
                    "official_split": official_split.get(study),
                    "reasons": ";".join(reasons),
                }
            )
            continue
        age = age_map.get(study, np.nan)
        samples.append(
            {
                "sample_id": f"{study}_{lateral}",
                "study_id": study,
                "laterality": lateral,
                "cc_image_id": c["_img"],
                "mlo_image_id": m["_img"],
                "cc_image_path": matches.at[c["_img"], "path"],
                "mlo_image_path": matches.at[m["_img"], "path"],
                "cc_image_relpath": matches.at[c["_img"], "relative_path"],
                "mlo_image_relpath": matches.at[m["_img"], "relative_path"],
                "cc_series_id": c[series] if series in df.columns else None,
                "mlo_series_id": m[series] if series in df.columns else None,
                "birads_original": c[cols["birads"]] if c["_bs"] == "ok" else None,
                "birads_encoded": c["_b"],
                "birads_valid": bool(c["_bs"] == "ok"),
                "density_original": c[cols["density"]] if c["_ds"] == "ok" else None,
                "density_encoded": c["_d"],
                "density_valid": bool(c["_ds"] == "ok"),
                "age": age,
                "age_missing_flag": int(pd.isna(age)),
                "official_split": official_split.get(study),
                "final_split": study_split.get(study),
            }
        )
    samples_df = pd.DataFrame(samples, columns=MANIFEST_COLUMNS)
    samples_df["birads_encoded"] = samples_df["birads_encoded"].astype("Int64")
    samples_df["density_encoded"] = samples_df["density_encoded"].astype("Int64")
    excluded_df = pd.DataFrame(excluded, columns=["study_id", "laterality", "n_records", "official_split", "reasons"])
    logger.info("Built %d breast samples, excluded %d", len(samples_df), len(excluded_df))
    return samples_df, excluded_df


def sample_distribution(samples: pd.DataFrame, birads_space: LabelSpace, density_space: LabelSpace) -> pd.DataFrame:
    """Per final split class counts of valid breast samples (used for ``split_distribution.csv``)."""
    rows = []
    for split_name, g in samples.groupby("final_split"):
        rows.append({"final_split": split_name, "task": "all", "class": "n_samples", "count": len(g)})
        for task, space, enc, valid in (
            ("birads", birads_space, "birads_encoded", "birads_valid"),
            ("density", density_space, "density_encoded", "density_valid"),
        ):
            vals = g.loc[g[valid], enc].astype(int)
            for k, name in enumerate(space.class_names):
                rows.append({"final_split": split_name, "task": task, "class": name, "count": int((vals == k).sum())})
    return pd.DataFrame(rows)
