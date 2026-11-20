"""PNG header parsing and bit-depth-preserving image loading (PNG only, no DICOM)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_TYPE_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
COLOR_TYPE_NAMES = {0: "grayscale", 2: "rgb", 3: "palette", 4: "grayscale_alpha", 6: "rgba"}


class PNGFormatError(ValueError):
    """Raised for PNG files that cannot be handled without losing information."""


def read_png_header(path: Path) -> dict[str, Any]:
    """Parse IHDR (width, height, bit depth, colour type) without decoding pixel data."""
    with open(path, "rb") as fh:
        head = fh.read(29)
    if len(head) < 29 or head[:8] != PNG_SIGNATURE:
        raise PNGFormatError(f"Not a valid PNG file: {path}")
    _, chunk_type = struct.unpack(">I4s", head[8:16])
    if chunk_type != b"IHDR":
        raise PNGFormatError(f"First PNG chunk is not IHDR: {path}")
    width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", head[16:29])
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "color_type_name": COLOR_TYPE_NAMES.get(color_type, "unknown"),
        "channels": COLOR_TYPE_CHANNELS.get(color_type),
        "interlace": interlace,
    }


def load_png_array(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a PNG as a 2-D ``uint8`` or ``uint16`` array, preserving the encoded bit depth.

    Three-channel files are reduced to one channel only if all channels are identical;
    otherwise ``PNGFormatError`` is raised. Palette, alpha and 16-bit multi-channel PNGs are rejected.
    """
    header = read_png_header(path)
    color_type, bit_depth = header["color_type"], header["bit_depth"]
    if color_type == 3:
        raise PNGFormatError(f"Palette PNG is not supported: {path}")
    if color_type in (4, 6):
        raise PNGFormatError(f"PNG with alpha channel is not supported: {path}")
    if bit_depth not in (8, 16):
        raise PNGFormatError(f"Unsupported PNG bit depth {bit_depth}: {path}")
    if color_type == 2 and bit_depth == 16:
        raise PNGFormatError(f"16-bit RGB PNG would lose bit depth in Pillow: {path}")
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        array = np.asarray(image)
    channels_identical = True
    if array.ndim == 3:
        if array.shape[2] != 3:
            raise PNGFormatError(f"Unexpected channel count {array.shape[2]}: {path}")
        channels_identical = bool(np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2]))
        if not channels_identical:
            raise PNGFormatError(f"RGB PNG with non-identical channels (unexpected colour information): {path}")
        array = array[..., 0]
    if array.dtype == np.int32:
        if array.min() < 0 or array.max() > 65535:
            raise PNGFormatError(f"Integer PNG values outside 16-bit range: {path}")
        array = array.astype(np.uint16)
    if array.dtype not in (np.uint8, np.uint16):
        raise PNGFormatError(f"Unsupported array dtype {array.dtype} for {path}")
    info = dict(header, pil_mode=mode, dtype=str(array.dtype), channels_identical=channels_identical)
    return np.ascontiguousarray(array), info


def dtype_max_value(array: np.ndarray) -> float:
    """Maximum representable value of the array's container dtype."""
    if array.dtype == np.uint8:
        return 255.0
    if array.dtype == np.uint16:
        return 65535.0
    raise PNGFormatError(f"Unsupported dtype {array.dtype}")
