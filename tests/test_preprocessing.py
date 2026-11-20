import numpy as np
import pandas as pd
import pytest
import torch

from sanvia.data.dataset import BreastPairDataset, PairedAugmenter
from sanvia.data.preprocessing import (
    AmbiguousIntensityRange,
    IntensityNormalizer,
    PNGPreprocessor,
    imagenet_normalize,
    letterbox,
    validate_normalization_against_audit,
)
from synthetic import write_png


def test_letterbox_preserves_aspect_ratio_and_does_not_crop():
    x = torch.ones(1, 100, 50)
    out, meta = letterbox(x, 64)
    assert out.shape == (1, 64, 64)
    h, w = meta["resized_hw"]
    assert h == 64 and w == 32 and abs(h / w - 100 / 50) < 0.05
    cols = (out[0].sum(0) > 0.5).nonzero().flatten()
    assert cols.min() == meta["pad_left"] and cols.max() == meta["pad_left"] + w - 1  # padding only, nothing cropped


def test_border_content_survives_resize():
    x = torch.zeros(1, 120, 80)
    x[:, :, 0] = 1.0
    x[:, :, -1] = 1.0
    out, meta = letterbox(x, 64)
    assert out[0, :, meta["pad_left"]].max() > 0.2 and out[0, :, meta["pad_left"] + meta["resized_hw"][1] - 1].max() > 0.2


def test_intensity_normalization_by_bit_depth():
    n = IntensityNormalizer("dtype_range")
    a8 = np.array([[0, 255]], dtype=np.uint8)
    a16 = np.array([[0, 65535, 32768]], dtype=np.uint16)
    assert n(a8).max() == 1.0 and n(a8).dtype == np.float32
    assert n(a16).max() == 1.0 and abs(n(a16)[0, 2] - 0.5) < 1e-4  # not divided by 255


def test_ambiguous_16bit_range_stops():
    audit = pd.DataFrame({"readable": [True], "stats_computed": [True], "bit_depth": [16], "max": [4095.0]})
    cfg = {"allowed_bit_depths": [8, 16], "ambiguous_16bit_max_threshold": 32768, "strategy": "dtype_range"}
    with pytest.raises(AmbiguousIntensityRange):
        validate_normalization_against_audit(audit, cfg)
    cfg["strategy"] = "global_window"
    assert validate_normalization_against_audit(audit, cfg)["bit_depths_found"] == [16]


def test_unsupported_bit_depth_stops():
    audit = pd.DataFrame({"readable": [True], "stats_computed": [False], "bit_depth": [1]})
    with pytest.raises(AmbiguousIntensityRange):
        validate_normalization_against_audit(
            audit, {"allowed_bit_depths": [8, 16], "ambiguous_16bit_max_threshold": 1, "strategy": "dtype_range"}
        )


def test_imagenet_normalization_values():
    x = torch.full((3, 4, 4), 0.5)
    out = imagenet_normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert torch.allclose(out[0], torch.tensor((0.5 - 0.485) / 0.229)) and torch.allclose(
        out[2], torch.tensor((0.5 - 0.406) / 0.225)
    )


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_preprocessor_shape_channels_dtype(tmp_path, base_cfg, bit_depth):
    write_png(tmp_path / "a.png", np.random.RandomState(0), size=(90, 60), bit_depth=bit_depth)
    pre = PNGPreprocessor(96, IntensityNormalizer(), base_cfg["data"]["preprocessing"])
    t = pre(tmp_path / "a.png")
    assert t.shape == (3, 96, 96) and t.dtype == torch.float32
    mean, std = base_cfg["data"]["preprocessing"]["imagenet_mean"], base_cfg["data"]["preprocessing"]["imagenet_std"]
    undone = [t[c] * std[c] + mean[c] for c in range(3)]
    assert torch.allclose(undone[0], undone[1], atol=1e-6) and torch.allclose(undone[1], undone[2], atol=1e-6)  # replicated gray
    assert pre.load_letterboxed(tmp_path / "a.png").max() <= 1.0


def test_dataset_item_and_masks(prepared):
    td, cfg = prepared["td"], prepared["cfg"]
    pre = PNGPreprocessor(64, td.normalizer, cfg["data"]["preprocessing"])
    df = td.train_df.copy()
    df.loc[0, ["density_valid"]] = False
    ds = BreastPairDataset(df, pre, PairedAugmenter(cfg["data"]["augmentation"]), np.zeros((len(df), 2), dtype=np.float32))
    item = ds[0]
    assert item["images"].shape == (2, 3, 64, 64) and not item["density_mask"] and item["birads_mask"]
    assert item["age"].shape == (2,)


def test_augmentation_keeps_range_and_background():
    aug = PairedAugmenter(
        {
            "rotation_degrees": 5.0,
            "translate_fraction": 0.03,
            "scale_range": [0.95, 1.05],
            "brightness_range": [0.92, 1.08],
            "contrast_range": [0.92, 1.08],
            "gamma_range": [0.92, 1.08],
            "gaussian_noise_std": 0.01,
            "p_geometric": 1.0,
            "p_photometric": 1.0,
            "p_noise": 1.0,
            "share_params_across_views": False,
        }
    )
    x = torch.zeros(1, 64, 64)
    x[:, 16:48, 16:48] = 0.6
    a, b = aug([x, x.clone()])
    assert a.min() >= 0 and a.max() <= 1 and a.shape == x.shape
    assert a[0, 0, 0] == 0 and not torch.equal(a, b)
