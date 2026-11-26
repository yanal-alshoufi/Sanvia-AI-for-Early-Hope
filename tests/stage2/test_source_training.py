from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset

from sanvia.stage2.source_model import SourceModel, masked_loss
from sanvia.stage2.source_training import plan_sources, run_stage
from sanvia.training.checkpointing import load_checkpoint
from sanvia.utils.seed import set_global_seed

ROOT=Path(__file__).resolve().parents[2]
SETTINGS={**yaml.safe_load((ROOT/"configs/stage2/default.yaml").read_text())["source_training"],"warmup_updates":1,
          "train_updates":2,"batch_size":2,"accumulation":2,"eval_every":2}
TASKS={"p":{"scope":"breast","classes":["a","b"],"loss":"ce","weight":1.0}}
class Backbone(nn.Sequential):
    feature_dim=3
    def __init__(self):
        super().__init__(nn.Conv2d(3,3,1),nn.BatchNorm2d(3),nn.AdaptiveAvgPool2d(1),nn.Flatten())


class Samples(Dataset):
    tasks=TASKS
    def __len__(self):return 5
    def __getitem__(self,i):
        if isinstance(i,tuple):i=i[0]
        return dict(images=torch.full((2,3,4,4),float(i)),view_mask=torch.tensor([True,i%2==0]),p=i%2,p_mask=True,sample_id=str(i))


@pytest.mark.parametrize("strategy",["sequential","multi-dataset"])
def test_update_boundary_resume(tmp_path,strategy):
    identity={"seed":7}
    ds={s:Samples() for s in ("A","B")}
    stages=plan_sources(list(ds),SETTINGS,strategy,7)
    assert Counter(s for _,steps,_ in stages for s,_,_ in steps)=={"A":3,"B":3}
    def train(path,resume,stop=None):
        set_global_seed(7)
        model=SourceModel(Backbone(),{s:TASKS for s in ds})
        for stage in stages:
            run_stage(model,ds,ds,{s:{} for s in ds},SETTINGS,torch.device("cpu"),stage,identity,path,resume,{},interrupt_after=stop)
        return model
    ref=train(tmp_path/"ref",False)
    folder=tmp_path/"resumed"
    with pytest.raises(InterruptedError):train(folder,False,(stages[0][0],1))
    restored=train(folder,True)
    for k,v in ref.state_dict().items():
        assert torch.equal(v,restored.state_dict()[k]),k
    last=load_checkpoint(folder/stages[-1][0]/"last_checkpoint.pt")
    assert last["accumulation_pending"]==0
    with pytest.raises(ValueError,match="resume"):train(folder,False)


def test_freezing_masking_and_invalid_batches():
    model=SourceModel(Backbone(),{"A":TASKS,"B":TASKS})
    x=torch.rand(2,2,3,4,4)
    mask=torch.tensor([[True,True],[True,False]])
    model.set_phase(True)
    before=model.backbone[1].running_mean.clone()
    logits=model(x,mask,"A")
    x[~mask]=1000
    assert torch.equal(logits["p"],model(x,mask,"A")["p"])
    assert torch.equal(before,model.backbone[1].running_mean)
    with pytest.raises(ValueError,match="valid image"):model(x,mask & False,"A")
    b=dict(p=torch.tensor([1,0]),p_mask=torch.tensor([True,False]))
    loss,_=masked_loss(logits,b,TASKS,{})
    loss.backward()
    assert all(p.grad is None for p in model.heads["B"].parameters())
    b["p_mask"][:]=False
    assert masked_loss(logits,b,TASKS,{})==(None,{})
    pred=torch.tensor([[1.,-3.]],requires_grad=True)
    b=dict(f=torch.ones_like(pred),f_mask=torch.tensor([[True,False]]))
    loss,_=masked_loss({"f":pred},b,{"f":{"loss":"bce","weight":1}},{});loss.backward()
    assert pred.grad[0,1]==0 and pred.grad[0,0]<0

