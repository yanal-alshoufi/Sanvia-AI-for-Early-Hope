"""Controlled VinDr training using the original factory, loaders, trainer and evaluators."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sanvia.data.pipeline import make_dataloaders
from sanvia.evaluation.evaluator import evaluate, measure_inference_time
from sanvia.evaluation.visualizations import plot_confusion_matrix, plot_learning_curves
from sanvia.models.multitask_model import build_model
from sanvia.stage2.export import export_backbone, load_export
from sanvia.stage2.stage1_artifacts import require
from sanvia.training.checkpointing import save_checkpoint, verify_resume_compatibility
from sanvia.training.losses import MultiTaskLoss
from sanvia.training.trainer import Trainer
from sanvia.utils.config import config_hash, run_config_subset
from sanvia.utils.io import save_json, sha256_file
from sanvia.utils.seed import set_global_seed


class TargetTrainer(Trainer):
    """Extend Stage 1 epoch checkpoints with the shuffle stream and complete Stage 2 identity."""
    def _payload(self,epoch):
        return dict(super()._payload(epoch),loader_rng=self.train_loader.generator.get_state())

    def _restore(self,ckpt):
        verify_resume_compatibility(ckpt,self.identity)
        self.train_loader.generator.set_state(ckpt["loader_rng"].cpu())
        start=super()._restore(ckpt)
        if ckpt["phase"]==2:
            self.early.load_state_dict(ckpt["early_stopping"])
            if self.early.should_stop:return self.f_epochs+self.t_epochs+1
        return start


def train_target(spec,snapshot,seed: int,pretrained: bool,source: Path | None,out: Path,identity: dict,resume: bool,device,tracker=None) -> dict:
    td=snapshot.data
    cfg=copy.deepcopy(snapshot.config)
    cfg["training"]["seed"]=seed
    t=cfg["training"]
    set_global_seed(seed,t["deterministic"])
    out.mkdir(parents=True,exist_ok=True)
    require(not (out/"last_checkpoint.pt").exists() or resume,"Target exists. Use resume or a new configuration.")
    # Identical RNG consumption for fusion/age/head initialization in all three arms.
    model=build_model(spec,t,td.age_enabled,pretrained=pretrained)
    initial=source or (out/"initial_backbone.pt")
    if source is not None:
        load_export(source,spec,model.backbone)
    else:
        if resume and initial.is_file():
            load_export(initial,spec,model.backbone)
        else:
            export_backbone(model.backbone,spec,initial,dict(identity=identity,origin="imagenet" if pretrained else "smoke_random",
                            preprocessing=cfg["data"]["preprocessing"]))
    protocol=config_hash(dict(config=run_config_subset(cfg,spec.name),dataset=td.dataset_manifest_hash,split=td.split_id,seed=seed,
                              normalizer=td.normalizer.to_dict(),class_weights=td.class_weights,age=td.age_processor.to_dict() if td.age_enabled else None,
                              samples=list(td.val_df.sample_id),code=identity["code"]),64)
    ckid=dict(config_hash=protocol,dataset_id=td.dataset_id,split_id=td.split_id,backbone_name=spec.name,dataset_manifest_hash=td.dataset_manifest_hash,
              initialization_sha256=sha256_file(initial),experiment_identity=identity)
    train,val=make_dataloaders(td,spec,cfg)
    w={k:torch.tensor(v,dtype=torch.float32) for k,v in td.class_weights.items()}
    criterion=MultiTaskLoss(w["birads"],w["density"],t["loss"]["task_weights"])
    trainer=TargetTrainer(model,cfg,train,val,criterion,device,out,ckid,dict(mlflow_run_id=tracker.run_id if tracker else None),tracker)
    trainer.last_path=out/"last_checkpoint.pt"
    result=trainer.fit(resume=resume)
    history=result.pop("history")
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    plot_learning_curves(history,t["phase1"]["epochs"],out/"learning_curves.png")
    amp=t["amp"] and device.type=="cuda"
    metrics=evaluate(model,val,criterion,device,amp,include_probabilities=True)
    preds=metrics.pop("predictions").merge(td.val_df[["sample_id","study_id"]],on="sample_id",how="left",validate="one_to_one")
    require(preds.study_id.notna().all() and len(preds)==len(td.val_df),"Validation sample mismatch")
    preds.to_csv(out/"validation_predictions.csv",index=False)
    for s in (td.birads_space,td.density_space):
        m=metrics[s.name]
        plot_confusion_matrix(m["confusion_matrix"],s.class_names,s.name,out/f"confusion_{s.name}.png")
        pd.DataFrame(m["per_class"],index=s.class_names).to_csv(out/f"per_class_{s.name}.csv")
        # Define minority classes from train counts, prior to target evaluation.
        counts=td.train_df.loc[td.train_df[f"{s.name}_valid"],f"{s.name}_encoded"].value_counts()
        minority=sorted(int(k) for k,v in counts.items() if v==counts.min())
        m["minority_classes"]=minority
        m["minority_recall"]=float(np.mean([m["per_class"]["recall"][k] for k in minority]))
    timing=t["inference_timing"]
    metrics["inference_ms_per_batch"]=measure_inference_time(model,spec.input_size,device,td.age_enabled,amp,timing["batch_size"],timing["warmup_iters"],timing["iters"])
    save_checkpoint(out/"vindr_best_model.pt",model.state_dict())
    meta=dict(identity=identity,protocol=protocol,initialization_sha256=ckid["initialization_sha256"],validation=metrics,
              origin="vindr_finetuned",preprocessing=cfg["data"]["preprocessing"])
    export_backbone(model.backbone,spec,out/"vindr_backbone.pt",meta)
    save_json(dict(meta,model_sha256=sha256_file(out/"vindr_best_model.pt"),architecture=t,age_enabled=td.age_enabled,
                   normalizer=td.normalizer.to_dict(),age_processor=td.age_processor.to_dict() if td.age_enabled else None),out/"vindr_model_metadata.json")
    summary=dict(identity=identity,backbone=spec.name,seed=seed,strategy=identity["strategy"],protocol=protocol,mode=identity["mode"],
                 initial=str(initial.resolve()),initial_sha256=ckid["initialization_sha256"],directory=str(out.resolve()),validation=metrics,
                 parameters=model.parameter_counts(),predictions_sha256=sha256_file(out/"validation_predictions.csv"),**result)
    save_json(summary,out/"target_summary.json")
    return summary

