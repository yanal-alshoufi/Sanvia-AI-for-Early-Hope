"""Paired validation comparisons, predefined independent selection and portable handoff."""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sanvia.evaluation.metrics import compute_task_metrics
from sanvia.stage2.export import load_export
from sanvia.stage2.stage1_artifacts import require
from sanvia.utils.io import load_json, save_json, sha256_file

TASKS={"birads":5,"density":4}
ARMS=("imagenet","sequential","multi-dataset")


def predictions(row):
    path=Path(row["directory"])/"validation_predictions.csv"
    require(sha256_file(path)==row["predictions_sha256"],"Validation predictions changed")
    return pd.read_csv(path,dtype={"study_id":str,"sample_id":str}).set_index("sample_id",verify_integrity=True).sort_index()


def paired_interval(a: pd.DataFrame,b: pd.DataFrame,iterations: int,seed: int) -> dict:
    """Percentile CI for paired mean-task F1 difference, resampling whole studies."""
    cols=["study_id",*[f"{k}_{suffix}" for k in TASKS for suffix in ("true","valid")]]
    require(a[cols].equals(b[cols]),"Paired predictions require identical sample identities, labels, masks and studies")
    groups=list(a.groupby("study_id",sort=True).indices.values())
    def score(frame,indices):
        scores=[]
        for t,n in TASKS.items():
            subset=frame.iloc[indices]
            subset=subset[subset[f"{t}_valid"]]
            truth=subset[f"{t}_true"].to_numpy(dtype=int)
            pred=subset[f"{t}_pred"].to_numpy(dtype=int)
            cm=np.bincount(n*truth+pred,minlength=n*n).reshape(n,n)
            supported=cm.sum(1)>0
            if not supported.any():return np.nan
            scores.append((2*np.diag(cm)[supported]/(cm.sum(0)+cm.sum(1))[supported]).mean())
        return np.mean(scores)
    rng=np.random.default_rng(seed)
    deltas=[]
    for _ in range(iterations):
        ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        delta=score(a,ix)-score(b,ix)
        if np.isfinite(delta):deltas.append(delta)
    return dict(iterations=iterations,valid=len(deltas),seed=seed,interval=np.quantile(deltas,[.025,.975]).tolist() if deltas else None)


def compare(rows: list[dict],pair: list[str],seeds: list[int],cfg: dict,out: Path,mode: str) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    index={(r["backbone"],r["seed"],r["strategy"]):r for r in rows}
    require(len(rows)==len(index) and set(index)=={(b,s,a) for b in pair for s in seeds for a in ARMS},"Incomplete or repeated paired experiment matrix")
    records=[]
    per_class=[]
    intervals={}
    for (name,seed,arm),r in index.items():
        baseline=index[name,seed,"imagenet"]
        require(r["mode"]==baseline["mode"]==mode and r["protocol"]==baseline["protocol"],"Incompatible target protocol")
        v,base=r["validation"],baseline["validation"]
        a,b=predictions(r),predictions(baseline)
        ci=paired_interval(a,b,cfg["bootstrap_iterations"] if arm!="imagenet" else 0,cfg["bootstrap_seed"])
        intervals[f"{name}/{seed}/{arm}"]=ci
        delta=v["selection_score"]-base["selection_score"]
        row=dict(backbone=name,seed=seed,strategy=arm,score=v["selection_score"],delta=delta,relative_gain=delta/base["selection_score"] if base["selection_score"] else None,
                 gap=r["train_val_f1_gap"],peak_gpu_mb=r["peak_gpu_mb"],training_time_s=r["training_time_s"])
        for t in TASKS:
            for metric in ("macro_f1","qwk","minority_recall"):
                value=v[t][metric]
                before=base[t][metric]
                row[f"{t}_{metric}"]=value
                row[f"{t}_{metric}_delta"]=value-before if value is not None and before is not None else None
            for k in range(TASKS[t]):
                for m in ("precision","recall","f1"):
                    value,before=v[t]["per_class"][m][k],base[t]["per_class"][m][k]
                    per_class.append(dict(backbone=name,seed=seed,strategy=arm,task=t,class_index=k,metric=m,value=value,
                                          delta=value-before if value is not None and before is not None else None))
        records.append(row)
    df=pd.DataFrame(records)
    df.to_csv(out/"seed_results.csv",index=False)
    pd.DataFrame(per_class).to_csv(out/"class_changes.csv",index=False)
    summary=df.groupby(["backbone","strategy"]).agg({"score":["mean","std"],"delta":["mean","std"],"training_time_s":"mean","peak_gpu_mb":"max"})
    summary.to_csv(out/"comparison.csv")
    save_json(intervals,out/"bootstrap.json")
    for field,title in [("score","Validation selection score"),("delta","Matched baseline improvement"),("training_time_s","Target training time (s)")]:
        ax=df.pivot_table(index="backbone",columns="strategy",values=field).plot.bar(rot=10)
        ax.set_ylabel(title);ax.figure.tight_layout();ax.figure.savefig(out/f"{field}.png");plt.close(ax.figure)
    decisions={}
    for name in pair:
        reasons={}
        candidates=[]
        for arm in ARMS[1:]:
            group=df[(df.backbone==name)&(df.strategy==arm)]
            blockers=[]
            if group.delta.mean()<cfg["min_gain"] or group.delta.isna().any():blockers.append("insufficient matched gain")
            if group.gap.isna().any() or group.gap.max()>cfg["max_overfit_gap"]:blockers.append("overfitting guard")
            for t in TASKS:
                for metric,limit in [("macro_f1","max_task_regression"),("qwk","max_qwk_regression"),("minority_recall","max_minority_recall_regression")]:
                    d=group[f"{t}_{metric}_delta"]
                    if d.isna().any() or d.min() < -cfg[limit]:blockers.append(f"{t} {metric} regression or undefined")
            if cfg["bootstrap_iterations"] and any((ci:=intervals[f"{name}/{s}/{arm}"]["interval"]) is None or ci[0]<=0 for s in seeds):
                blockers.append("paired interval includes non-improvement")
            reasons[arm]=blockers
            if not blockers:
                candidates.append((-group.score.mean(),group.score.std(ddof=0),ARMS.index(arm),arm))
        chosen=min(candidates)[-1] if candidates else "imagenet"
        group=df[(df.backbone==name)&(df.strategy==chosen)]
        seed=int(min(zip(abs(group.score-group.score.mean()),group.seed))[1])
        decisions[name]=dict(strategy=chosen,representative_seed=seed,guards=reasons)
    result=dict(mode=mode,eligible_for_handoff=mode=="full",decisions=decisions,official_test_used=False,policy=cfg)
    save_json(result,out/"selection_report.json")
    return result


def handoff(rows: list[dict],report: dict,lineage: dict,out: Path) -> None:
    require(report["eligible_for_handoff"] and report["mode"]=="full","Only a completed full benchmark can create a research handoff")
    choices={}
    for name,d in report["decisions"].items():
        row=next(r for r in rows if (r["backbone"],r["seed"],r["strategy"])==(name,d["representative_seed"],d["strategy"]))
        source=Path(row["initial"])
        require(sha256_file(source)==row["initial_sha256"],"Selected pre-target artifact has changed")
        dest=out/name
        dest.mkdir(parents=True,exist_ok=True)
        path=dest/"backbone.pt"
        shutil.copyfile(source,path)
        meta=load_json(source.with_name(source.stem+"_metadata.json"))
        meta.update(file=path.name,target_validation=row["validation"],selection=d)
        save_json(meta,dest/"backbone_metadata.json")
        load_export(path)
        save_json(meta["spec"],dest/"backbone_config.json")
        (dest/"model_card.md").write_text(f"# {name}\n\nResearch artifact. This model has not been clinically validated.\n\nInitialization: {d['strategy']}. Representative seed: {d['representative_seed']}. Validation score: {row['validation']['selection_score']:.5f}. See selection_report.json for thresholds and lineage_report.json for training provenance.\n")
        choices[name]=dict(file=f"{name}/backbone.pt",sha256=meta["sha256"],**d)
    save_json(choices,out/"selected_initializations.json")
    save_json(report,out/"selection_report.json")
    save_json(lineage,out/"lineage_report.json")
    save_json(dict(lineage,official_test_used=False),out/"reproducibility_report.json")
    (out/"README.md").write_text("# Research handoff\n\nInstall the recorded SANVIA revision. Pass each backbone.pt path to sanvia.stage2.export.load_export. The loader verifies the checksum and rebuilds the original feature extractor. All files in this directory are required.\n")

