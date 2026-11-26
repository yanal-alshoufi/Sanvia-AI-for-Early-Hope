from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch

from sanvia.models.backbone_factory import build_backbone, get_backbone_spec
from sanvia.stage2.comparison import paired_interval
from sanvia.stage2.export import export_backbone, load_export
from sanvia.stage2.stage1_artifacts import read_snapshot
from sanvia.utils.config import config_hash, run_config_subset, save_resolved_config
from sanvia.utils.io import save_json


def test_export_reload_and_tampering(base_cfg,tmp_path):
    spec=next(get_backbone_spec(base_cfg,b["name"]) for b in base_cfg["backbones"] if b["builder"]=="mobilenet_v3_large")
    model=build_backbone(spec,False)
    path=tmp_path/"source.pt"
    export_backbone(model,spec,path,{"origin":"synthetic"})
    other=load_export(path,spec)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    with path.open("ab") as f:f.write(b"x")
    with pytest.raises(ValueError,match="checksum"):load_export(path,spec)


def test_paired_groups():
    df=pd.DataFrame(dict(study_id=["a","a","b"],birads_true=[0,0,1],birads_pred=[0,1,1],birads_valid=[True]*3,
                    density_true=[1,1,2],density_pred=[1,2,2],density_valid=[True]*3))
    assert paired_interval(df,df,30,72)["interval"]==[0,0]
    with pytest.raises(ValueError,match="identical"):
        paired_interval(df,df.iloc[:2],5,2)


def test_snapshot_reads_stage1_layout(prepared):
    cfg,paths,td=prepared["cfg"],prepared["paths"],prepared["td"]
    names=[b["name"] for b in cfg["backbones"][-2:]]
    decision={"selected":names,"basis":"validation only"}
    runs=[]
    for name in names:
        spec=get_backbone_spec(cfg,name)
        run_id=name+"-synthetic-fixture"
        loc=paths.checkpoints_dir/name/run_id
        loc.mkdir(parents=True)
        save_resolved_config(cfg,loc/"resolved_config.yaml")
        summary=dict(backbone=name,run_id=run_id,dataset_id=td.dataset_id,split_id=td.split_id,input_size=spec.input_size,
                     weights_identifier=f"{spec.weights_enum}.{spec.weights_name}",config_hash=config_hash(run_config_subset(cfg,name)),
                     git_commit="fixture",validation={"selection_score":0.125})
        save_json(summary,loc/"run_summary.json")
        runs.append(dict(backbone=name,run_id=run_id,validation_selection_score=.125))
    report=dict(official_test_used=False,final_selection=decision,config_hash=config_hash(cfg),dataset=prepared["samples"]["dataset_record"],git={"commit":"fixture"},experiments=runs)
    save_json(report,paths.reports_dir/"reproducibility_report.json")
    save_json(decision,paths.reports_dir/"selection_decision.json")
    save_resolved_config(cfg,paths.reports_dir/"resolved_config.yaml")
    snapshot=read_snapshot(paths.output_dir,paths.images_dir)
    assert snapshot.selected==names
    assert all(v["incompatibilities"] for v in snapshot.baselines.values())

