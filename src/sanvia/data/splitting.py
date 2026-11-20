"""Official split verification and study-level Train/Validation split (official Training only)."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sanvia.data.labels import LabelSpace, encode_series
from sanvia.utils.io import sha256_text

logger = logging.getLogger(__name__)


def verify_official_split(breast: pd.DataFrame, cols: Mapping[str, str], train_value: str, test_value: str) -> dict[str, Any]:
    """Check split values and that each study belongs to exactly one official split."""
    sid, split = cols["study_id"], cols["split"]
    values = set(breast[split].dropna().astype(str).str.strip())
    expected = {train_value, test_value}
    if values != expected:
        raise ValueError(
            f"Unexpected official split values {sorted(values)}; expected {sorted(expected)}. "
            "Update data.split.official_*_value after inspecting the CSV."
        )
    per_study = breast.groupby(sid)[split].nunique()
    straddling = per_study[per_study > 1].index.tolist()
    if straddling:
        raise ValueError(f"{len(straddling)} studies appear in both official splits, e.g. {straddling[:5]}")
    counts = breast.groupby(split)[sid].nunique().to_dict()
    return {
        "split_values": sorted(values),
        "studies_per_official_split": counts,
        "expected_note": "Approx. 4000 training and 1000 test studies expected; actual counts reported.",
    }


def study_signatures(
    breast: pd.DataFrame, cols: Mapping[str, str], birads_space: LabelSpace, density_space: LabelSpace
) -> pd.DataFrame:
    """One row per study with the maximum valid BI-RADS and density code among its breasts."""
    df = breast[[cols["study_id"], cols["split"]]].copy()
    df["b"], _ = encode_series(breast[cols["birads"]], birads_space)
    df["d"], _ = encode_series(breast[cols["density"]], density_space)
    grouped = (
        df.groupby(cols["study_id"])
        .agg(official_split=(cols["split"], "first"), max_birads=("b", "max"), max_density=("d", "max"))
        .reset_index()
    )
    grouped = grouped.rename(columns={cols["study_id"]: "study_id"})
    fmt = lambda v: "NA" if pd.isna(v) else str(int(v))  # noqa: E731
    grouped["sig_full"] = [f"B{fmt(b)}_D{fmt(d)}" for b, d in zip(grouped["max_birads"], grouped["max_density"])]
    grouped["sig_birads"] = [f"B{fmt(b)}" for b in grouped["max_birads"]]
    return grouped


def choose_stratification(studies: pd.DataFrame, min_stratum: int, val_fraction: float = 0.2) -> tuple[pd.Series | None, str]:
    """Finest label signature that supports a stratified split; rare strata fall back to BI-RADS only.

    A candidate is usable only if every stratum has at least 2 studies and the validation set can hold
    at least one study per stratum (required by stratified splitting).
    """
    n_val = int(np.ceil(val_fraction * len(studies)))

    def usable(labels: pd.Series) -> bool:
        counts = labels.value_counts()
        return counts.min() >= 2 and len(counts) <= n_val and len(counts) <= len(studies) - n_val

    full = studies["sig_full"].copy()
    counts = full.value_counts()
    rare = counts[counts < min_stratum].index
    merged = full.where(~full.isin(rare), studies["sig_birads"] + "_rare")
    if usable(merged):
        return merged, "birads_density_signature_with_rare_merge"
    if usable(studies["sig_birads"]):
        return studies["sig_birads"], "birads_only"
    return None, "none_random_split"


def split_train_validation(
    studies: pd.DataFrame,
    val_fraction: float,
    seed: int,
    stratify: bool,
    min_stratum: int,
) -> tuple[list[str], list[str], str]:
    """Split official-Training studies into Train and Validation by ``study_id``."""
    studies = studies.sort_values("study_id").reset_index(drop=True)
    labels, level = (None, "disabled")
    if stratify:
        labels, level = choose_stratification(studies, min_stratum, val_fraction)
    ids = studies["study_id"].astype(str).tolist()
    train_ids, val_ids = train_test_split(
        ids, test_size=val_fraction, random_state=seed, stratify=None if labels is None else labels.values
    )
    return sorted(train_ids), sorted(val_ids), level


def compute_split_id(
    seed: int, val_fraction: float, level: str, train_ids: list[str], val_ids: list[str], test_ids: list[str]
) -> str:
    payload = (
        f"seed={seed}|val={val_fraction}|level={level}|"
        + ",".join(train_ids)
        + "|"
        + ",".join(val_ids)
        + "|"
        + ",".join(test_ids)
    )
    return "split-" + sha256_text(payload)[:12]


def build_split(
    breast: pd.DataFrame,
    cols: Mapping[str, str],
    split_cfg: Mapping[str, Any],
    birads_space: LabelSpace,
    density_space: LabelSpace,
) -> dict[str, Any]:
    """Create Train/Validation from official Training only and keep official Test unchanged."""
    verification = verify_official_split(breast, cols, split_cfg["official_train_value"], split_cfg["official_test_value"])
    studies = study_signatures(breast, cols, birads_space, density_space)
    studies["study_id"] = studies["study_id"].astype(str)
    studies["official_split"] = studies["official_split"].astype(str).str.strip()
    dev = studies[studies["official_split"] == split_cfg["official_train_value"]]
    test = studies[studies["official_split"] == split_cfg["official_test_value"]]
    train_ids, val_ids, level = split_train_validation(
        dev, split_cfg["val_fraction"], split_cfg["seed"], split_cfg["stratify"], split_cfg["min_stratum_size"]
    )
    test_ids = sorted(test["study_id"].tolist())
    studies["final_split"] = studies["study_id"].map(
        {**{i: "train" for i in train_ids}, **{i: "validation" for i in val_ids}, **{i: "official_test" for i in test_ids}}
    )
    result = {
        "studies": studies,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "stratification_level": level,
        "verification": verification,
        "split_id": compute_split_id(split_cfg["seed"], split_cfg["val_fraction"], level, train_ids, val_ids, test_ids),
    }
    assert_no_leakage(result)
    return result


def assert_no_leakage(split: Mapping[str, Any]) -> None:
    """Raise if any study appears in more than one final split."""
    tr, va, te = set(split["train_ids"]), set(split["val_ids"]), set(split["test_ids"])
    for name, overlap in (("train/validation", tr & va), ("train/test", tr & te), ("validation/test", va & te)):
        if overlap:
            raise AssertionError(f"Study leakage between {name}: {sorted(overlap)[:5]}")


def split_summary(split: Mapping[str, Any], split_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split_id": split["split_id"],
        "seed": split_cfg["seed"],
        "val_fraction": split_cfg["val_fraction"],
        "stratification_level": split["stratification_level"],
        "splitting_unit": "study_id",
        "n_studies": {
            "train": len(split["train_ids"]),
            "validation": len(split["val_ids"]),
            "official_test": len(split["test_ids"]),
        },
        "expected_n_studies_note": "Expected approx. 3200 / 800 / 1000; actual counts are reported.",
        "split_config": dict(split_cfg),
        "official_split_verification": split["verification"],
    }


def study_label_distribution(studies: pd.DataFrame) -> pd.DataFrame:
    """Study-level signature distribution per final split (used to inspect stratification quality)."""
    t = studies.groupby(["final_split", "sig_birads"]).size().rename("n_studies").reset_index()
    t["fraction_within_split"] = t["n_studies"] / t.groupby("final_split")["n_studies"].transform("sum")
    return t
