"""Synthetic native-format source collections, deliberately unrelated to clinical images."""

from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import DigitalMammographyXRayImageStorageForPresentation, ExplicitVRLittleEndian


def projection(path: Path, patient: str, n: int, view="CC", mono="MONOCHROME2"):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DigitalMammographyXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = f"2.25.{n + 1}"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=bytes(128))
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = f"2.25.{n + 10000}"
    ds.SeriesInstanceUID = f"2.25.{n + 20000}"
    ds.PatientID = patient
    ds.ImageLaterality = "L"
    ds.ViewPosition = view
    ds.Rows, ds.Columns = 16, 8
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.SamplesPerPixel, ds.PixelRepresentation = 1, 0
    ds.PhotometricInterpretation = mono
    ds.PixelData = (np.random.default_rng(n).integers(1, 4095, size=(16, 8), dtype="<u2")).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return ds


def make_sources(root: Path):
    configs = {}
    for i, name in enumerate(("CBIS-DDSM", "CMMD", "INbreast")):
        source = root / name
        rows = []
        for j in range(8):
            patient = f"P_{j:05d}"
            file = f"{j}_scan.dcm"
            projection(source / file, patient, 1000 * i + j)
            if i == 0:
                rows.append(
                    dict(
                        patient_id=patient,
                        **{
                            "left or right breast": "LEFT",
                            "image view": "CC",
                            "abnormality id": "1",
                            "abnormality type": "mass",
                            "image file path": file,
                            "cropped image file path": f"crop{j}",
                            "ROI mask file path": f"mask{j}",
                            "breast_density": str(1 + j % 4),
                            "pathology": ["BENIGN", "MALIGNANT"][j % 2],
                        },
                    )
                )
            if i == 1:
                rows.append(dict(ID1=patient, LeftRight="L", classification=["Benign", "Malignant"][j % 2]))
            if i == 2:
                rows.append(
                    {
                        "File Name": str(j),
                        "Laterality": "L",
                        "View": "CC",
                        "Bi-Rads": ["0", "4a", "4b", "6"][j % 4],
                        "ACR": str(1 + j % 4),
                    }
                )
        if i == 0:
            rows.append({**rows[0], "abnormality id": "2", "abnormality type": "calcification", "pathology": "MALIGNANT"})
        file = "clinical.xlsx" if i == 1 else "annotations.csv"
        if i == 1:
            pd.DataFrame(rows).to_excel(source / file, index=False)
        else:
            pd.DataFrame(rows).to_csv(source / file, sep=";" if i == 2 else ",", index=False)
        configs[name] = dict(root=str(source), version="synthetic-v1", annotations=[dict(path=file, official_split="train")])
    return configs


def smoke_inputs(root: Path, repo: Path, cfg: dict):
    """Construct a small CPU-only fixture with two explicitly designated development backbones."""
    from PIL import Image

    from sanvia.data.labels import build_label_spaces
    from sanvia.data.pipeline import TrainingData
    from sanvia.data.preprocessing import IntensityNormalizer
    from sanvia.stage2.stage1_artifacts import Snapshot
    from sanvia.training.losses import compute_class_weights
    from sanvia.utils.config import config_hash, load_config
    from sanvia.utils.io import load_json, save_json, sha256_file

    root.mkdir(parents=True, exist_ok=True)
    marker = root / "sources.json"
    if not marker.exists():
        save_json(make_sources(root), marker)
    cfg["sources"] = load_json(marker)
    config = load_config(repo / "configs")
    config["training"].update(batch_size=4, effective_batch_size=4, num_workers=0, amp=False)
    config["training"]["phase1"]["epochs"] = 1
    config["training"]["phase2"].update(epochs=1, unfreeze_epochs=1)
    config["training"]["inference_timing"].update(batch_size=2, warmup_iters=0, iters=1)
    config["training"]["fusion"]["hidden_dim"] = 16
    config["data"]["augmentation"]["enabled"] = False
    names = ["MobileNetV3-Large", "DenseNet121"]
    for b in config["backbones"]:
        b["input_size"] = 64
    rows = []
    hashes = set()
    for i in range(16):
        row = dict(
            study_id=f"synthetic-study-{i // 2}",
            sample_id=f"synthetic-breast-{i}",
            final_split="train" if i < 10 else "validation",
            official_split="training",
            birads_encoded=i % 5,
            density_encoded=i % 4,
            birads_valid=True,
            density_valid=True,
        )
        for v, view in enumerate(("cc", "mlo")):
            file = root / f"target-{i}-{view}.png"
            if not file.exists():
                Image.fromarray(np.random.default_rng(2000 + 2 * i + v).integers(0, 256, (64, 32), dtype=np.uint8)).save(file)
            hashes.add(sha256_file(file))
            row[f"{view}_image_path"] = str(file)
        rows.append(row)
    frame = pd.DataFrame(rows)
    train = frame[frame.final_split == "train"].reset_index(drop=True)
    val = frame[frame.final_split == "validation"].reset_index(drop=True)
    spaces = build_label_spaces(config)
    weights = {s.name: compute_class_weights(train[f"{s.name}_encoded"].to_numpy(), s.num_classes) for s in spaces}
    td = TrainingData(
        train,
        val,
        *spaces,
        IntensityNormalizer(),
        weights,
        False,
        None,
        "Synthetic fixture without age",
        config_hash(sorted(hashes)),
        config_hash(rows),
        config_hash(train.sample_id.tolist()),
    )
    return Snapshot(config, names, td, {}, {"git": {"commit": "synthetic-fixture"}, "official_test_used": False}, hashes)
