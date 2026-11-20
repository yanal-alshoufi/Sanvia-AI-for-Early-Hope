import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sanvia.data.image_indexing import find_unexpected_files, index_png_files, match_annotations_to_images
from sanvia.data.labels import build_label_spaces, encode_series, encode_value
from sanvia.data.loading import build_schema_report, normalize_id, read_csv_checked, validate_required_columns
from sanvia.data.png_io import PNGFormatError, load_png_array, read_png_header
from synthetic import write_png


def test_read_csv_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata.csv"):
        read_csv_checked(tmp_path / "nope.csv", "metadata.csv")


def test_required_column_validation():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_required_columns(df, ["a", "b"], "t")


def test_schema_report_fields(tmp_path):
    df = pd.DataFrame({"x": ["a", "b", "a"], "y": [1, 2, None]})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    rep = build_schema_report({"t": df}, {"t": p})["t"]
    assert rep["shape"] == [3, 2] and rep["missing_values"]["y"] == 1 and rep["categorical_unique_values"]["x"] == ["a", "b"]


def test_indexing_duplicates_missing_unexpected(tmp_path):
    rng = np.random.RandomState(0)
    write_png(tmp_path / "a" / "img1.png", rng)
    write_png(tmp_path / "b" / "img1.png", rng)  # duplicate stem
    write_png(tmp_path / "a" / "img2.png", rng)
    write_png(tmp_path / "a" / "extra.png", rng)  # unexpected
    idx = index_png_files(tmp_path)
    assert set(idx.duplicates) == {"img1"}
    m = match_annotations_to_images(["img1.png", "img2", "gone"], idx).set_index("image_id")
    assert m.loc["img1", "status"] == "ambiguous" and m.loc["img2", "status"] == "ok" and m.loc["gone", "status"] == "missing"
    assert find_unexpected_files(idx, {"img1", "img2"})["image_id"].tolist() == ["extra"]


def test_no_substring_matching(tmp_path):
    write_png(tmp_path / "abc123.png", np.random.RandomState(0))
    m = match_annotations_to_images(["abc"], index_png_files(tmp_path))
    assert m.loc[0, "status"] == "missing"
    assert normalize_id(" x.PNG ") == "x"


def test_png_bit_depth_preserved(tmp_path):
    rng = np.random.RandomState(0)
    write_png(tmp_path / "8.png", rng, bit_depth=8)
    write_png(tmp_path / "16.png", rng, bit_depth=16)
    a8, i8 = load_png_array(tmp_path / "8.png")
    a16, i16 = load_png_array(tmp_path / "16.png")
    assert a8.dtype == np.uint8 and i8["bit_depth"] == 8
    assert a16.dtype == np.uint16 and i16["bit_depth"] == 16 and a16.max() > 255
    assert read_png_header(tmp_path / "16.png")["width"] == 64


def test_rgb_identical_channels_reduced_and_color_rejected(tmp_path):
    g = (np.random.RandomState(0).rand(10, 12) * 255).astype(np.uint8)
    Image.fromarray(np.stack([g, g, g], -1)).save(tmp_path / "gray_rgb.png")
    arr, info = load_png_array(tmp_path / "gray_rgb.png")
    assert arr.ndim == 2 and info["channels_identical"]
    color = np.stack([g, g[::-1], g], -1)
    Image.fromarray(color).save(tmp_path / "color.png")
    with pytest.raises(PNGFormatError):
        load_png_array(tmp_path / "color.png")


def test_corrupted_png_detected(tmp_path):
    write_png(tmp_path / "ok.png", np.random.RandomState(0))
    data = (tmp_path / "ok.png").read_bytes()
    (tmp_path / "bad.png").write_bytes(data[:60])
    with pytest.raises(Exception):
        load_png_array(tmp_path / "bad.png")


def test_label_encoding_exact_and_unknown(base_cfg):
    birads, density = build_label_spaces(base_cfg)
    assert encode_value("BI-RADS 4", birads) == (3, "ok")
    assert encode_value(" density d ", density) == (3, "ok")
    assert encode_value("BI-RADS 6", birads) == (None, "unknown")
    assert encode_value("category 4 extra", birads)[1] == "unknown"  # no regex extraction
    assert encode_value(float("nan"), birads) == (None, "missing")
    enc, st = encode_series(pd.Series(["BI-RADS 1", None, "x"]), birads)
    assert list(st) == ["ok", "missing", "unknown"] and enc.isna().tolist() == [False, True, True]
