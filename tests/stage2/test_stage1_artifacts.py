from pathlib import Path

import pandas as pd
import pytest

from sanvia.stage2.stage1_artifacts import read_target
from sanvia.utils.io import load_json


def test_exact_target_split_without_official_test_images(prepared):
    paths = prepared["paths"]
    rows = prepared["samples"]["samples"]
    for name in ("cc", "mlo"):
        for file in rows.loc[rows.final_split == "official_test", f"{name}_image_path"]:
            Path(file).unlink()
    dataset = load_json(paths.manifests_dir / "dataset_manifest.json")
    data, hashes = read_target(paths.output_dir, paths.images_dir, prepared["cfg"], dataset)
    assert set(data.train_df.sample_id) == set(prepared["td"].train_df.sample_id)
    assert set(data.val_df.sample_id) == set(prepared["td"].val_df.sample_id)
    assert hashes and data.age_enabled and data.split_id == dataset["split_id"]
    path = paths.splits_dir / "train_studies.csv"
    studies = pd.read_csv(path)
    studies.loc[0, "final_split"] = "validation"
    studies.to_csv(path, index=False)
    with pytest.raises(ValueError, match="partition"):
        read_target(paths.output_dir, paths.images_dir, prepared["cfg"], dataset)
