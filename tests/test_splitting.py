import pandas as pd
import pytest

from sanvia.data.labels import build_label_spaces
from sanvia.data.splitting import assert_no_leakage, build_split, verify_official_split

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


def _split(cfg, paths):
    b, d = build_label_spaces(cfg)
    breast = pd.read_csv(paths.breast_annotations_csv)
    return breast, build_split(breast, COLS, cfg["data"]["split"], b, d)


def test_split_is_study_level_and_leak_free(synth):
    cfg, paths = synth
    breast, s = _split(cfg, paths)
    tr, va, te = set(s["train_ids"]), set(s["val_ids"]), set(s["test_ids"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) + len(va) == 30 and len(va) == 6 and len(te) == 10
    official_test = set(breast.loc[breast.split == "test", "study_id"])
    assert te == official_test and not (official_test & (tr | va))  # official Test never feeds Train/Validation
    per_study = s["studies"].groupby("study_id")["final_split"].nunique()
    assert per_study.eq(1).all()
    assert_no_leakage(s)


def test_split_is_reproducible_and_id_stable(synth):
    cfg, paths = synth
    _, a = _split(cfg, paths)
    _, b = _split(cfg, paths)
    assert a["split_id"] == b["split_id"] and a["train_ids"] == b["train_ids"]


def test_both_breasts_of_study_share_split(prepared):
    s = prepared["samples"]["samples"]
    assert s.groupby("study_id")["final_split"].nunique().eq(1).all()
    assert set(s.loc[s.official_split == "test", "final_split"]) == {"official_test"}


def test_unexpected_split_values_raise(synth):
    cfg, paths = synth
    breast = pd.read_csv(paths.breast_annotations_csv)
    breast.loc[0, "split"] = "validation"
    with pytest.raises(ValueError, match="Unexpected official split values"):
        verify_official_split(breast, COLS, "training", "test")


def test_study_in_both_official_splits_raises(synth):
    cfg, paths = synth
    breast = pd.read_csv(paths.breast_annotations_csv)
    breast.loc[breast.index[0], "split"] = "test"
    with pytest.raises(ValueError, match="both official splits"):
        verify_official_split(breast, COLS, "training", "test")


def test_leakage_assertion_triggers():
    with pytest.raises(AssertionError):
        assert_no_leakage({"train_ids": ["a"], "val_ids": ["a"], "test_ids": []})
