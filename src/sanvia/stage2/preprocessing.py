"""Conservative external-source intensity and geometry preprocessing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut

from sanvia.data.preprocessing import letterbox
from sanvia.stage2.stage1_artifacts import require
from sanvia.utils.config import config_hash
from sanvia.utils.io import sha256_file, sha256_text


def pixels(path: Path) -> tuple[np.ndarray, dict]:
    """Decode a single grayscale projection and exclude stored padding from the DICOM range."""
    dicom = path.suffix.lower() in {".dcm",".dicom",""}
    if dicom:
        ds = pydicom.dcmread(path)
        photo = str(ds.PhotometricInterpretation)
        shape = str(ds.get("PresentationLUTShape","")).upper()
        require(photo in {"MONOCHROME1","MONOCHROME2"} and shape in {"","IDENTITY","INVERSE"} and
                "PresentationLUTSequence" not in ds, f"Unsupported DICOM presentation in {path}")
        require(str(ds.get("PresentationIntentType","FOR PRESENTATION")) != "FOR PROCESSING",f"Processing-intent DICOM needs a verified presentation transform: {path}")
        raw = ds.pixel_array
        require(raw.ndim == 2,f"Expected one grayscale frame: {path}")
        valid = np.ones(raw.shape,dtype=bool)
        if "PixelPaddingValue" in ds:
            a,b=sorted((float(ds.PixelPaddingValue),float(ds.get("PixelPaddingRangeLimit",ds.PixelPaddingValue))))
            valid=(raw<a)|(raw>b)
        require(valid.any(),f"All pixels are padding: {path}")
        x=np.asarray(apply_voi_lut(apply_modality_lut(raw,ds),ds),dtype=np.float64)
        low,high=x[valid].min(),x[valid].max()
        require(np.isfinite([low,high]).all() and high>low,f"No usable intensity range: {path}")
        x=np.clip((x-low)/(high-low),0.,1.)
        # An explicit presentation LUT supplies the transform to P-values.
        if shape=="INVERSE" or (not shape and photo=="MONOCHROME1"):
            x=1-x
        x[~valid]=0
        bits=int(ds.BitsStored)
    else:
        with Image.open(path) as img:
            raw=np.asarray(img)
        if raw.ndim==3 and raw.shape[-1]==3 and np.array_equal(raw[:,:,0],raw[:,:,1]) and np.array_equal(raw[:,:,0],raw[:,:,2]):
            raw=raw[:,:,0]
        require(raw.ndim==2 and raw.dtype.kind=="u",f"Expected unsigned grayscale mammogram: {path}")
        bits=raw.dtype.itemsize*8
        x=raw.astype(np.float32)/np.iinfo(raw.dtype).max
    return x.astype(np.float32),dict(width=x.shape[1],height=x.shape[0],bit_depth=bits,image_format="DICOM" if dicom else path.suffix[1:].upper())


def preprocess(path: Path, digest: str, size: int, config: dict, cache: Path | None = None) -> torch.Tensor:
    """Cache deterministic letterboxed floats only. Validate cache and input before reuse."""
    require(sha256_file(path)==digest,f"Source changed after audit: {path}")
    key=config_hash(dict(version=1,file=digest,size=size,config=config),64)
    def build():
        x,_=pixels(path)
        return letterbox(torch.from_numpy(x)[None],size,config["interpolation"],config["antialias"],config["pad_value"])[0]
    if cache is None:
        return build()
    cache.mkdir(parents=True,exist_ok=True)
    target=cache/(key+".npz")
    if target.exists():
        with np.load(target,allow_pickle=False) as data:
            array=data["image"]
            require(array.shape==(1,size,size) and array.dtype==np.float32 and data["digest"].item()==array_digest(array,key),f"Corrupt cache entry {target}")
            return torch.from_numpy(array)
    result=build()
    with tempfile.NamedTemporaryFile(dir=cache,suffix=".npz",delete=False) as tmp:
        np.savez(tmp,image=result.numpy(),digest=array_digest(result.numpy(),key))
    os.replace(tmp.name,target)
    return result


def array_digest(array: np.ndarray, key: str) -> str:
    import hashlib
    return sha256_text(key+hashlib.sha256(array.tobytes()).hexdigest())

