import pandas as pd

from sanvia.data.image_indexing import index_png_files, match_annotations_to_images
from sanvia.data.labels import build_label_spaces
from sanvia.data.sample_builder import build_breast_samples

COLS = {
    "study_id": "study_id",
    "series_id": "series_id",
    "image_id": "image_id",
    "laterality": "laterality",
    "view_position": "view_position",
    "birads": "breast_birads",
    "density": "breast_density",
    "split": "split",
}


def _build(cfg, paths, breast, corrupted=frozenset()):
    idx = index_png_files(paths.images_dir)
    match = match_annotations_to_images(breast["image_id"], idx)
    b, d = build_label_spaces(cfg)
    studies = breast["study_id"].unique()
    return build_breast_samples(
        breast,
        match,
        set(corrupted),
        COLS,
        b,
        d,
        ["L", "R"],
        {"cc": "CC", "mlo": "MLO"},
        {s: "train" for s in studies},
        {s: "training" for s in studies},
        None,
    )


def _breast(paths):
    return pd.read_csv(paths.breast_annotations_csv)


def test_one_sample_per_breast_with_cc_mlo_pair(synth):
    cfg, paths = synth
    samples, excluded = _build(cfg, paths, _breast(paths))
    assert len(samples) == 80 and excluded.empty
    r = samples.iloc[0]
    assert r["sample_id"] == f"{r['study_id']}_{r['laterality']}"
    assert r["cc_image_id"].endswith("CC") and r["mlo_image_id"].endswith("MLO")
    assert r["cc_image_id"][3:6] == r["mlo_image_id"][3:6] and r["cc_image_id"][6] == r["mlo_image_id"][6]
    assert samples.groupby("study_id").size().eq(2).all()


def test_missing_view_and_duplicates_excluded_not_substituted(synth):
    cfg, paths = synth
    b = _breast(paths)
    b = b[~((b.study_id == "study000") & (b.laterality == "L") & (b.view_position == "MLO"))]
    dup = b[(b.study_id == "study001") & (b.laterality == "R") & (b.view_position == "CC")].copy()
    b = pd.concat([b, dup])
    samples, excluded = _build(cfg, paths, b)
    reasons = dict(zip(excluded.study_id + "_" + excluded.laterality, excluded.reasons))
    assert "missing_mlo" in reasons["study000_L"] and "duplicate_cc_records" in reasons["study001_R"]
    assert "study000_L" not in set(samples.sample_id) and "study001_R" not in set(samples.sample_id)


def test_conflicting_labels_flagged_not_resolved(synth):
    cfg, paths = synth
    b = _breast(paths)
    mask = (b.study_id == "study002") & (b.laterality == "L") & (b.view_position == "MLO")
    b.loc[mask, "breast_birads"] = "BI-RADS 5"
    b.loc[(b.study_id == "study002") & (b.laterality == "L") & (b.view_position == "CC"), "breast_birads"] = "BI-RADS 1"
    samples, excluded = _build(cfg, paths, b)
    assert (
        "birads_conflict_between_views"
        in excluded.set_index(excluded.study_id + "_" + excluded.laterality).loc["study002_L", "reasons"]
    )


def test_missing_label_is_masked_not_zero(synth):
    cfg, paths = synth
    b = _breast(paths)
    b.loc[(b.study_id == "study003") & (b.laterality == "R"), "breast_density"] = None
    b.loc[(b.study_id == "study004") & (b.laterality == "R"), ["breast_density", "breast_birads"]] = None
    samples, excluded = _build(cfg, paths, b)
    s = samples.set_index("sample_id")
    assert (
        not s.loc["study003_R", "density_valid"]
        and pd.isna(s.loc["study003_R", "density_encoded"])
        and s.loc["study003_R", "birads_valid"]
    )
    assert "study004_R" not in s.index and "no_valid_target" in excluded.reasons.iloc[0]


def test_unknown_label_reported_and_excluded(synth):
    cfg, paths = synth
    b = _breast(paths)
    b.loc[(b.study_id == "study005") & (b.laterality == "L"), "breast_birads"] = "BI-RADS 9"
    samples, excluded = _build(cfg, paths, b)
    assert "birads_unknown_label" in excluded.reasons.iloc[0]


def test_unreadable_image_excluded(synth):
    cfg, paths = synth
    samples, excluded = _build(cfg, paths, _breast(paths), corrupted={"img006LCC"})
    assert "cc_image_unreadable" in excluded.reasons.iloc[0]
