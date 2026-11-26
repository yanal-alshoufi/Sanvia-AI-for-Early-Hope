from pathlib import Path

import numpy as np
import pytest
import yaml

from sanvia.stage2.audit import assign_splits, audit_sources, encode
from sanvia.stage2.preprocessing import pixels, preprocess
from sanvia.stage2.synthetic import make_sources, projection
from sanvia.utils.io import sha256_file

ROOT = Path(__file__).resolve().parents[2]
TASKS = yaml.safe_load((ROOT / "configs/stage2/tasks.yaml").read_text())


def test_real_adapters_on_synthetic_sources(tmp_path):
    cfg = dict(sources=make_sources(tmp_path), enabled_sources=list(TASKS), split_seed=8, validation_fraction=0.25)
    result = audit_sources(cfg, TASKS, tmp_path / "report", set())
    cbis = result["CBIS-DDSM"]
    assert len(cbis["samples"]) == 8 and len(cbis["images"]) == 8
    s = cbis["samples"][0]
    assert s["views"][1] is None and s["tasks"]["pathology"][0]["mask"] is False
    assert s["tasks"]["finding"][0]["mask"] == [True, True]
    assert cbis["samples"][1]["tasks"]["finding"][0]["mask"] == [True, False]
    assert all(s["tasks"]["pathology"][0]["mask"] for s in result["CMMD"]["samples"])
    assert result["INbreast"]["samples"][0]["tasks"]["birads"][0]["y"] == 0
    assert {s["split"] for s in cbis["samples"]} == {"train", "validation"}
    h = next(iter(cbis["images"].values()))["sha256"]
    with pytest.raises(ValueError, match="cross-dataset"):
        audit_sources(cfg, TASKS, tmp_path / "conflict", {h})


def test_invalid_vocab_and_patient_overlap():
    with pytest.raises(ValueError, match="vocabulary"):
        encode([{"birads": "4?"}], TASKS["INbreast"]["birads"], "unanimous")
    sample = dict(patient_id="x", eligible=True, tasks={})
    with pytest.raises(ValueError, match="crosses"):
        assign_splits([dict(sample, official_split="test"), dict(sample, official_split="train")], 42, 0.2)


def test_intensity_polarity_and_cache(tmp_path):
    img = tmp_path / "view.dcm"
    ds = projection(img, "synthetic", 2)
    a, meta = pixels(img)
    assert meta["bit_depth"] == 12 and a.dtype == np.float32
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.PresentationLUTShape = "INVERSE"
    ds.save_as(img, enforce_file_format=True)
    b, _ = pixels(img)
    np.testing.assert_allclose(a + b, 1)
    cfg = dict(interpolation="bilinear", antialias=True, pad_value=0)
    digest = sha256_file(img)
    out = preprocess(img, digest, 32, cfg, tmp_path / "cache")
    assert out.shape == (1, 32, 32) and out[:, :, :8].count_nonzero() == 0
    assert (out == preprocess(img, digest, 32, cfg, tmp_path / "cache")).all()
    img.write_bytes(b"broken")
    with pytest.raises(ValueError, match="changed"):
        preprocess(img, digest, 32, cfg, tmp_path / "cache")
    with pytest.raises(Exception):
        pixels(img)


@pytest.mark.parametrize("name", ["BCDR-FM", "BCDR-DM"])
def test_release_bound_adapter_and_derivative_exclusion(tmp_path, name):
    import pandas as pd
    from PIL import Image

    from sanvia.stage2.audit import build_source

    root = tmp_path / name
    root.mkdir()
    records = []
    for j in range(5):
        file = f"{j}.tiff"
        Image.fromarray(np.random.default_rng(j).integers(0, 65535, (17, 9), dtype=np.uint16)).save(root / file)
        records.append(
            dict(
                subject=str(j),
                exam=str(j),
                id=str(j),
                side="L",
                view="CC",
                file=file,
                type="mask" if j == 4 else "full",
                diagnosis=["Benign", "Malignant"][j % 2],
            )
        )
    table = root / "fixture.csv"
    pd.DataFrame(records).to_csv(table, index=False)
    columns = dict(
        patient="subject",
        study="exam",
        image="id",
        side="side",
        view="view",
        path="file",
        image_type="type",
        pathology="diagnosis",
    )
    cfg = dict(
        root=root,
        annotations=[dict(path=table.name, columns=columns)],
        schema_reference="fixture data dictionary",
        version="test-release",
        tasks=TASKS["CMMD"],
    )
    dataset = build_source(name, cfg, cfg["tasks"], tmp_path / "audit", 11, 0.25)
    assert len(dataset["images"]) == 4 and all(i["bit_depth"] == 16 for i in dataset["images"].values())
    assert set(s["patient_id"].split("/")[0] for s in dataset["samples"]) == {name}
    assert pd.read_csv(tmp_path / "audit/excluded_samples.csv").iloc[0]["reason"] == "mask"


def test_schema_and_projection_disagreements(tmp_path):
    import pandas as pd

    from sanvia.stage2.adapters import CBIS, CMMD

    configs = make_sources(tmp_path)
    info = configs["CBIS-DDSM"]
    path = Path(info["root"]) / info["annotations"][0]["path"]
    df = pd.read_csv(path, dtype=str)
    df.loc[0, "left or right breast"] = "RIGHT"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="laterality"):
        CBIS(info).records()
    info = configs["CMMD"]
    path = Path(info["root"]) / info["annotations"][0]["path"]
    pd.DataFrame({"ID1": ["SYNTH"]}).to_excel(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        CMMD(info).records()
