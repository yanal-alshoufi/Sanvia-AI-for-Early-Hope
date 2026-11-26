"""Deterministic source schedules, checkpoint selection, and optimizer-boundary recovery."""
from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sanvia.evaluation.evaluator import autocast_context, batch_to_device
from sanvia.evaluation.metrics import compute_task_metrics
from sanvia.models.backbone_factory import build_backbone
from sanvia.stage2.export import export_backbone, load_export
from sanvia.stage2.source_data import SourceDataset, draws, loader
from sanvia.stage2.source_model import SourceModel, masked_loss
from sanvia.stage2.stage1_artifacts import require
from sanvia.training.checkpointing import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from sanvia.utils.config import config_hash
from sanvia.utils.io import load_json, save_json, sha256_file
from sanvia.utils.seed import set_global_seed


def plan_sources(names: list[str],settings: dict,strategy: str,seed: int) -> list:
    """Each tuple specifies source, head-only warm-up, and optional replay."""
    quotas={s:int(settings["quotas"].get(s,settings["train_updates"])) for s in names}
    warm=settings["warmup_updates"]
    require(all(v>0 for v in quotas.values()) and warm>=0,"Source update quotas must be positive")
    rng=np.random.default_rng(seed)
    stages=[]
    if strategy=="sequential":
        for i,s in enumerate(names):
            steps=[(s,True,False)]*warm
            for j in range(quotas[s]):
                steps.append((s,False,False))
                if i and int((j+1)*settings["replay_fraction"])>int(j*settings["replay_fraction"]):
                    steps.append((str(rng.choice(names[:i])),False,True))
            stages.append((s,steps,names[:i+1]))
    else:
        require(strategy=="multi-dataset" and not settings["replay_fraction"],"Replay is a sequential-only variant")
        steps=[]
        for frozen in (True,False):
            pool=[(s,frozen,False) for s in names for _ in range(warm if frozen else quotas[s])]
            rng.shuffle(pool)
            steps.extend(pool)
        stages.append(("joint",steps,names))
    return stages


@torch.no_grad()
def source_scores(model, datasets: dict,names: list[str],settings: dict,device) -> dict:
    """Equal weight per evaluable CE task within a dataset. Positive-only BCE has recall only."""
    model.eval()
    results={}
    for s in names:
        ds=datasets[s]
        values={k:[] for k in ds.tasks}
        for data in loader(ds,settings):
            b=batch_to_device(data,device)
            out=model(b["images"],b["view_mask"],s)
            for k,t in ds.tasks.items():
                mask=b[k+"_mask"]
                if mask.any():
                    values[k].append((b[k][mask].cpu(),out[k][mask].float().cpu()))
        metrics={}
        ranked=[]
        for k,t in ds.tasks.items():
            if not values[k]:
                metrics[k]=dict(n=0,score=None)
                continue
            truth,logits=[torch.cat(v) for v in zip(*values[k])]
            if t["loss"]=="ce":
                prob=logits.softmax(-1).numpy()
                m=compute_task_metrics(truth.numpy(),prob.argmax(-1),prob,len(t["classes"]))
                metrics[k]=m
                ranked.append(m["macro_f1"])
            else:
                metrics[k]=dict(positive_recall=float((logits>=0).float().mean()),known_positive_count=len(truth))
        require(bool(ranked),f"No evaluable multiclass validation task for {s}; review split and registry")
        results[s]=dict(tasks=metrics,score=float(np.mean(ranked)))
    return results


def run_stage(model,train,val,weights,settings,device,stage,identity,out,resume,origin,tracker=None,interrupt_after=None):
    """Save model, optimizer, RNG and per-source draw positions after every committed update."""
    name,steps,seen=stage
    folder=out/name
    folder.mkdir(parents=True,exist_ok=True)
    last,best=folder/"last_checkpoint.pt",folder/"best.pt"
    ckpt=load_checkpoint(last) if last.exists() else None
    if ckpt:
        require(resume and ckpt["identity"]==identity and ckpt["stage"]==name,"Incompatible or unrequested source resume")
        model.load_state_dict(ckpt["model"])
    pos=ckpt["step"] if ckpt else 0
    counts=Counter(ckpt["draws"] if ckpt else {})
    history=ckpt["history"] if ckpt else []
    best_score=ckpt["best_score"] if ckpt else -math.inf
    best_step=ckpt["best_step"] if ckpt else 0
    active_phase=None
    amp=bool(settings["amp"] and device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    windows=settings["accumulation"]
    per_update=settings["batch_size"]*windows
    sources=list(train)
    indices=Counter(s for s,_,_ in steps[pos:])
    streams={}
    for s,n in indices.items():
        seed=identity["seed"]+1009*sources.index(s)
        stream=draws(len(train[s]),counts[s],per_update*n,seed)
        streams[s]=iter(loader(train[s],settings,stream,seed))
    if ckpt:
        restore_rng_state(ckpt["rng"])
    for step in range(pos,len(steps)):
        s,frozen,replay=steps[step]
        if active_phase!=frozen:
            active_phase=frozen
            optimizer=torch.optim.AdamW([{"params":model.heads.parameters(),"lr":settings["frozen_head_lr"] if frozen else settings["head_lr"]},
                                         {"params":model.backbone.parameters(),"lr":settings["backbone_lr"]}],weight_decay=settings["weight_decay"])
            duration=sum(f==frozen for _,f,_ in steps)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=duration)
            if ckpt and ckpt["frozen"]==frozen and step==pos:
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                scaler.load_state_dict(ckpt["scaler"])
        model.set_phase(frozen)
        start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        ids=[]
        task_losses=Counter()
        task_counts=Counter()
        parts_count=Counter()
        effective=0
        views=0
        total_loss=0.
        for _ in range(windows):
            b=batch_to_device(next(streams[s]),device)
            ids.extend(b["sample_id"])
            views+=int(b["view_mask"].sum())
            with autocast_context(device,amp):
                outputs=model(b["images"],b["view_mask"],s)
                loss,parts=masked_loss(outputs,b,train[s].tasks,weights[s])
            if loss is None: continue
            for k,v in parts.items():
                task_losses[k]+=v
                parts_count[k]+=1
                task_counts[k]+=int(b[k+"_mask"].sum())
            total_loss+=float(loss.detach())
            scaler.scale(loss).backward()
            effective+=1
        require(effective>0,"No supervised microbatches in a planned optimizer update")
        scaler.unscale_(optimizer)
        for p in model.parameters():
            if p.grad is not None: p.grad.div_(effective)
        if origin and not frozen:
            for k,p in model.backbone.named_parameters():
                if p.grad is not None:
                    p.grad.add_(p.detach()-origin[k],alpha=2*settings["l2sp"])
        torch.nn.utils.clip_grad_norm_(model.parameters(),settings["clip_norm"],error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        counts[s]+=len(ids)
        rec=dict(step=step+1,source=s,frozen=frozen,replay=replay,loss=total_loss/effective,samples=len(ids),images=views,
                 sample_ids=ids,valid_labels=dict(task_counts),task_losses={k:v/parts_count[k] for k,v in task_losses.items()},
                 seconds=time.perf_counter()-start)
        if (step+1)%settings["eval_every"]==0 or step+1==len(steps):
            scores=source_scores(model,val,seen,settings,device)
            rec["validation"]={k:v["score"] for k,v in scores.items()}
            score=rec["validation"][name] if name!="joint" else float(np.mean(list(rec["validation"].values())))
            if score>best_score:
                best_score,best_step=score,step+1
                save_checkpoint(folder/f"best-{best_step}.pt",dict(identity=identity,model=model.state_dict(),validation=scores,step=best_step))
        history.append(rec)
        if tracker:tracker.log_metrics({f"{name}.source_loss":rec["loss"]},step+1)
        save_checkpoint(last,dict(identity=identity,stage=name,step=step+1,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                                  scaler=scaler.state_dict(),rng=capture_rng_state(),draws=dict(counts),best_score=best_score,best_step=best_step,
                                  frozen=frozen,history=history,mlflow_run_id=tracker.run_id if tracker else None,accumulation_pending=0))
        # The committed last checkpoint names its best. An interrupted replacement
        # cannot invalidate its previous best before this commit.
        for old in folder.glob("best-*.pt"):
            if old.name!=f"best-{best_step}.pt":old.unlink()
        if interrupt_after==(name,step+1):raise InterruptedError("Injected optimizer-boundary interruption")
    chosen=folder/f"best-{best_step}.pt"
    selected=load_checkpoint(chosen)
    require(selected["identity"]==identity and selected["step"]==best_step,"Invalid source-stage best checkpoint")
    model.load_state_dict(selected["model"])
    # Small filesystem pointer avoids duplicating a full optimizer checkpoint.
    save_json(dict(file=chosen.name,sha256=sha256_file(chosen),step=best_step),best.with_suffix(".json"))
    return history,selected["validation"]


def train_source(spec,snapshot,data,cfg,settings,strategy,seed,out: Path,identity: dict,resume: bool,device,tracker=None,interrupt_after=None):
    """Run a complete strategy for one backbone, starting from the Stage 1 weight enum."""
    set_global_seed(seed)
    out.mkdir(parents=True,exist_ok=True)
    registry={s:d["tasks"] for s,d in data.items()}
    model=SourceModel(build_backbone(spec,pretrained=cfg["pretrained"]),registry).to(device)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
    pre=snapshot.config["data"]["preprocessing"]
    cache=Path(cfg["cache"]) if cfg["cache"] else None
    def datasets(split):
        return {s:SourceDataset(d,Path(cfg["sources"][s]["root"]),split,spec.input_size,pre,snapshot.config["data"]["augmentation"],cache) for s,d in data.items()}
    train,val=datasets("train"),datasets("validation")
    weights={s:ds.weights(device) for s,ds in train.items()}
    origin={k:p.detach().clone() for k,p in model.backbone.named_parameters()} if settings["l2sp"] else {}
    stages=plan_sources(list(data),settings,strategy,seed)
    budget=[]
    for stage,steps,_ in stages:
        for (s,warm,replay),n in Counter(steps).items():
            samples=n*settings["batch_size"]*settings["accumulation"]
            budget.append(dict(stage=stage,source=s,warmup=warm,replay=replay,planned_updates=n,planned_samples=samples,
                               available_train=len(train[s]),expected_passes=samples/len(train[s])))
    pd.DataFrame(budget).to_csv(out/"budget.csv",index=False)
    history=[]
    matrix={}
    for stage in stages:
        rows,scores=run_stage(model,train,val,weights,settings,device,stage,identity,out,resume,origin,tracker,interrupt_after)
        history.extend(dict(r,stage=stage[0]) for r in rows)
        matrix[stage[0]]={s:m["score"] for s,m in scores.items()}
        save_json(scores,out/stage[0]/"best_validation.json")
    save_json(history,out/"source_history.json")
    df=pd.DataFrame(history)
    df.drop(columns=["sample_ids"]).to_csv(out/"source_history.csv",index=False)
    exposure={}
    for s,g in df.groupby("source"):
        ids=sum(g.sample_ids.tolist(),[])
        exposure[s]=dict(updates=len(g),samples=len(ids),images=int(g.images.sum()),unique_samples=len(set(ids)),repeats=len(ids)-len(set(ids)),fraction=len(g)/len(df))
    save_json(exposure,out/"actual_budget.json")
    scores=pd.DataFrame.from_dict(matrix,orient="index")
    scores.to_csv(out/"forgetting_matrix.csv")
    save_json((scores.max()-scores.iloc[-1]).to_dict(),out/"forgetting.json")
    fig,ax=plt.subplots();plot=ax.imshow(scores.to_numpy(),vmin=0,vmax=1,aspect="auto");fig.colorbar(plot,ax=ax)
    ax.set_xticks(range(len(scores.columns)),scores.columns);ax.set_yticks(range(len(scores.index)),scores.index)
    fig.tight_layout();fig.savefig(out/"forgetting.png");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3));axes[0].plot(df.loss);axes[0].set_title("Source loss")
    pd.DataFrame([r.get("validation",{}) for r in history]).plot(ax=axes[1],style="o");axes[1].set_title("Validation macro F1")
    axes[2].bar(list(exposure),[r["fraction"] for r in exposure.values()]);axes[2].set_title("Update fraction")
    fig.tight_layout();fig.savefig(out/"source_training.png");plt.close(fig)
    path=out/"source_adapted_backbone.pt"
    meta=export_backbone(model.backbone,spec,path,dict(identity=identity,origin="external_supervised",tasks=registry,preprocessing=pre,budget=budget,
                        selected_by="current-source or equal-dataset mean of per-source task means",peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0))
    load_export(path,spec)
    save_json(dict(backbone_sha256=meta["sha256"],identity=identity),out/"completed.json")
    return path

