from pathlib import Path

import pytest
import torch
import torchvision
from torch import nn

from sanvia.stage2.cli import smoke_config
from sanvia.stage2.comparison import compare, handoff
from sanvia.stage2.export import load_export
from sanvia.stage2.pipeline import Benchmark, configuration
from sanvia.stage2.tracking import stage_dvc, tracked
from sanvia.utils.io import load_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]


class FixtureNetwork(nn.Module):
    """A cheap deterministic stand-in for the six-arm orchestration regression."""

    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = nn.Linear(3, dim)
        self.classifier = nn.Identity()

    def forward(self, x):
        return self.classifier(self.features(self.pool(x).flatten(1)))


def test_primary_entry_points_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(torchvision.models, "mobilenet_v3_large", lambda weights: FixtureNetwork(960))
    monkeypatch.setattr(torchvision.models, "densenet121", lambda weights: FixtureNetwork(1024))
    bench = Benchmark(ROOT, configuration(ROOT, smoke_config(tmp_path)))
    out = bench.execute("run-all")
    results = [load_json(p) for p in out.glob("*/*/*/target/target_summary.json")]
    assert len(results) == 6 and len(list(out.glob("*/*/*/source/source_adapted_backbone.pt"))) == 4
    assert all(len({r["protocol"] for r in results if r["backbone"] == b}) == 1 for b in bench.snapshot.selected)
    track = next(out.glob("*/*/*/source/*/last_checkpoint.pt"))
    before = track.stat().st_mtime_ns
    reopened = Benchmark(ROOT, configuration(ROOT, smoke_config(tmp_path)))
    reopened.execute("run-all")
    assert reopened.out == out and track.stat().st_mtime_ns == before
    report = load_json(out / "comparison/selection_report.json")
    assert report["eligible_for_handoff"] is False
    with pytest.raises(ValueError, match="full"):
        handoff(results, report, {}, out / "forbidden")
    # Construct a controlled decision-policy fixture from the six rows. No
    # numbers or exports from this temporary test are research results.
    first, second = bench.snapshot.selected
    for r in results:
        r["mode"] = "full"
        r["train_val_f1_gap"] = 0.01
        score = 0.7 if r["backbone"] == first and r["strategy"] == "sequential" else 0.5
        r["validation"]["selection_score"] = score
        for t in ("birads", "density"):
            r["validation"][t].update(macro_f1=score, qwk=0.5, minority_recall=0.5)
    chosen = compare(results, [first, second], [42], bench.cfg["selection"], out / "policy-fixture", "full")
    assert chosen["decisions"][first]["strategy"] == "sequential"
    assert chosen["decisions"][second]["strategy"] == "imagenet"
    dest = out / "handoff-fixture"
    handoff(results, chosen, {}, dest)
    init = load_json(dest / "selected_initializations.json")
    for value in init.values():
        load_export(dest / value["file"])
    bench.execute("linear-probe")
    assert len(list(out.glob("*/*/*/linear_probe/target_summary.json"))) == 6


def test_separate_tracking_and_dvc(tmp_path):
    import mlflow

    identity = dict(benchmark="fixture", backbone="fixture", strategy="sequential", seed=42)
    folder = tmp_path / "arm" / "source"
    with tracked(tmp_path, folder, identity, True) as tracker:
        tracker.log_metrics({"validation": 0.25}, 1)
    rid = load_json(folder / "mlflow_run.json")["run_id"]
    assert mlflow.get_run(rid).data.metrics["validation"] == 0.25
    with tracked(tmp_path, folder, identity, True) as reused:
        assert reused.run_id == rid
    assert mlflow.get_run(rid).data.tags["mlflow.parentRunId"]
    marker = tmp_path / "hashes.json"
    marker.write_text('{"fixture":"checksum"}')
    status = stage_dvc(ROOT, tmp_path, {"hashes.json": marker})
    assert (tmp_path / "tracking/dvc_workspace" / status["files"][0]).read_bytes() == marker.read_bytes()
    assert not status["images_backed_up"]


def test_mode_and_source_gates():
    for overrides in ({"mode": "full", "pretrained": False}, {"mode": "pilot", "synthetic": True}, {"enabled_sources": ["CMMD"]}):
        with pytest.raises(ValueError):
            configuration(ROOT, overrides)
