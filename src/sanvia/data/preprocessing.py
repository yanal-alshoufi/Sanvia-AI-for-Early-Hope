"""PNG preprocessing: audit-driven intensity normalisation, letterboxing and ImageNet normalisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sanvia.data.png_io import dtype_max_value, load_png_array

logger = logging.getLogger(__name__)


class AmbiguousIntensityRange(RuntimeError):
    """Raised when the effective intensity range of the PNGs cannot be determined safely."""


@dataclass
class IntensityNormalizer:
    """Map raw PNG values to ``[0, 1]``.

    ``dtype_range``: divide by the container maximum (255 for uint8, 65535 for uint16), verified by the audit.
    ``global_window``: dtype scaling followed by one global window ``[lower, upper]`` estimated on Train only.
    """

    strategy: str = "dtype_range"
    lower: float | None = None
    upper: float | None = None

    def __call__(self, array: np.ndarray) -> np.ndarray:
        x = array.astype(np.float32) / np.float32(dtype_max_value(array))
        if self.strategy == "global_window":
            if self.lower is None or self.upper is None:
                raise RuntimeError("global_window normaliser has not been fitted on Train.")
            x = np.clip((x - self.lower) / (self.upper - self.lower), 0.0, 1.0)
        elif self.strategy != "dtype_range":
            raise ValueError(f"Unknown normalisation strategy: {self.strategy}")
        return x.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "lower": self.lower, "upper": self.upper}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "IntensityNormalizer":
        return cls(d["strategy"], d.get("lower"), d.get("upper"))


def validate_normalization_against_audit(image_audit: pd.DataFrame, norm_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Stop with a diagnostic report if bit depths or observed ranges make the mapping ambiguous."""
    readable = image_audit[image_audit["readable"]]
    allowed = set(norm_cfg["allowed_bit_depths"])
    depths = set(int(v) for v in readable["bit_depth"].dropna().unique())
    report: dict[str, Any] = {"bit_depths_found": sorted(depths), "strategy": norm_cfg["strategy"]}
    if not depths <= allowed:
        raise AmbiguousIntensityRange(f"Bit depths {sorted(depths - allowed)} are not supported (allowed: {sorted(allowed)}).")
    stats = readable[readable["stats_computed"]] if "stats_computed" in readable else readable.iloc[0:0]
    for depth in depths:
        sub = stats[stats["bit_depth"] == depth]
        if sub.empty:
            continue
        observed_max = float(sub["max"].max())
        report[f"observed_max_{depth}bit"] = observed_max
        if observed_max > (2**depth - 1):
            raise AmbiguousIntensityRange(f"Observed max {observed_max} exceeds the {depth}-bit range.")
        if depth == 16 and norm_cfg["strategy"] == "dtype_range" and observed_max < norm_cfg["ambiguous_16bit_max_threshold"]:
            raise AmbiguousIntensityRange(
                f"16-bit PNGs have a global max of {observed_max}, far below 65535: the effective range is ambiguous. "
                "Inspect the image statistics (image_audit.csv) and set data.normalization.strategy to global_window "
                "explicitly if this is intended."
            )
    return report


def fit_global_window(paths: Sequence[Path], norm_cfg: Mapping[str, Any], seed: int) -> IntensityNormalizer:
    """Estimate one global window from Train images only (percentiles of non-zero scaled intensities)."""
    gw = norm_cfg["global_window"]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(paths), min(gw["n_images"], len(paths)), replace=False)
    base = IntensityNormalizer("dtype_range")
    values = []
    for i in chosen:
        arr, _ = load_png_array(Path(paths[i]))
        x = base(arr).ravel()[::16]
        values.append(x[x > 0])
    pooled = np.concatenate(values)
    lo, hi = np.percentile(pooled, [gw["lower_percentile"], gw["upper_percentile"]])
    if hi <= lo:
        raise AmbiguousIntensityRange(f"Degenerate global window [{lo}, {hi}].")
    return IntensityNormalizer("global_window", float(lo), float(hi))


def letterbox(
    x: torch.Tensor, size: int, interpolation: str = "bilinear", antialias: bool = True, pad_value: float = 0.0
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resize ``[1, H, W]`` to fit a ``size x size`` square, preserving aspect ratio, and centre-pad."""
    _, h, w = x.shape
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    y = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode=interpolation, align_corners=False, antialias=antialias).squeeze(
        0
    )
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out = torch.full((1, size, size), pad_value, dtype=x.dtype)
    out[:, top : top + new_h, left : left + new_w] = y
    return out, {"scale": scale, "resized_hw": (new_h, new_w), "pad_top": top, "pad_left": left}


def replicate_channels(x: torch.Tensor) -> torch.Tensor:
    """Grayscale ``[1, H, W]`` to three identical channels ``[3, H, W]``."""
    return x.expand(3, -1, -1).contiguous()


def imagenet_normalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    return (x - m) / s


class PNGPreprocessor:
    """Deterministic preprocessing for one input resolution (no cropping, no distortion)."""

    def __init__(self, input_size: int, normalizer: IntensityNormalizer, pre_cfg: Mapping[str, Any]):
        self.input_size = input_size
        self.normalizer = normalizer
        self.mean, self.std = pre_cfg["imagenet_mean"], pre_cfg["imagenet_std"]
        self.interpolation, self.antialias, self.pad_value = pre_cfg["interpolation"], pre_cfg["antialias"], pre_cfg["pad_value"]

    def load_letterboxed(self, path: Path) -> torch.Tensor:
        """Load PNG, scale intensities to [0, 1], letterbox to ``[1, S, S]``."""
        array, _ = load_png_array(Path(path))
        x = torch.from_numpy(self.normalizer(array)).unsqueeze(0)
        boxed, _ = letterbox(x, self.input_size, self.interpolation, self.antialias, self.pad_value)
        return boxed

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        """Replicate to three channels and apply ImageNet normalisation: ``[3, S, S]`` float32."""
        return imagenet_normalize(replicate_channels(x), self.mean, self.std)

    def __call__(self, path: Path) -> torch.Tensor:
        return self.finalize(self.load_letterboxed(path))


def plot_preprocessing_examples(
    samples: pd.DataFrame, preprocessor: PNGPreprocessor, out_path: Path, n: int = 3
) -> dict[str, Any]:
    """Save original / processed CC and MLO figures and return tensor shape and value ranges."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = samples.head(n)
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows)), squeeze=False)
    info: dict[str, Any] = {}
    for i, (_, r) in enumerate(rows.iterrows()):
        for j, view in enumerate(("cc", "mlo")):
            raw, hdr = load_png_array(Path(r[f"{view}_image_path"]))
            final = preprocessor(Path(r[f"{view}_image_path"]))
            axes[i, 2 * j].imshow(raw, cmap="gray")
            axes[i, 2 * j].set_title(f"{view.upper()} original {raw.dtype} [{raw.min()}, {raw.max()}]", fontsize=8)
            axes[i, 2 * j + 1].imshow(final[0].numpy(), cmap="gray")
            axes[i, 2 * j + 1].set_title(
                f"{view.upper()} processed {tuple(final.shape)} [{final.min():.2f}, {final.max():.2f}]", fontsize=8
            )
            for ax in axes[i, 2 * j : 2 * j + 2]:
                ax.axis("off")
            info[f"{r['sample_id']}_{view}"] = {
                "raw_dtype": str(raw.dtype),
                "raw_range": [float(raw.min()), float(raw.max())],
                "tensor_shape": list(final.shape),
                "final_range": [float(final.min()), float(final.max())],
            }
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return info
