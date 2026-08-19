#!/usr/bin/env python3
"""Inference-side, no-retrain measurement on the frozen deployment model:
  - baseline    : single checkpoint, raw
  - TTA         : average softmax over the physics-consistent dihedral views the model
                  trained on (polar H/V flip = channel swap [0,3,2,1]; rot90/270 = [2,3,0,1])
  - soup        : uniform average of the 3 pfc4 seed weights, one deployable model
  - soup + TTA  : both
Reports factory miss@FA5 / miss@FA10 on Aug26/val for each. Baseline single seeds: s42=10.28,
s1=10.19, s2=10.44.
"""
import json, glob, sys
import numpy as np
import torch

from Config import load_hyperparameters
from Datasets import (build_val_only, clean_class_index, load_training_dataset,
                      val_dataloader)
from Metrics import miss_at_fa_sweep
from LitClassifier import Classifier

CFG = 'pol_pfc4_s42.json'
CKPTS = {s: glob.glob(f'/home/ammar/smoke_h5/overnight/pol_pfc4_s{s}/*.ckpt')[0] for s in ('42','1','2')}
# physics-consistent dihedral views (applied to the raw uint8 tile, as augment_train_batch does)
def views(x):
    yield x
    yield x.flip(-1)[:, [0,3,2,1]]                         # H mirror
    yield x.flip(-2)[:, [0,3,2,1]]                         # V mirror
    yield torch.rot90(x, 1, dims=(-2,-1))[:, [2,3,0,1]]    # +90
    yield torch.rot90(x, 3, dims=(-2,-1))[:, [2,3,0,1]]    # -90

def build_model(cfg, state):
    # Shared class-space loader -- the trainer's own call, so these indices are the ones
    # the checkpoints predict in.
    ds=load_training_dataset(cfg, label='ema', verbose=False)
    classes=list(ds.classes)
    if hasattr(ds,'file'): ds.file.close()
    clean_id=clean_class_index(classes)
    # Single source of truth (LitClassifier.Classifier.from_config). This block omitted
    # Max/Min/RangePolarization -- each of which adds an input channel, so those runs died
    # in the stem -- and the whole CustomCNN ladder. pretrained/seed_pretrained_stem stay
    # forced off: the state dict below overwrites everything, so a download buys nothing.
    m=Classifier.from_config(cfg,num_classes=len(classes),clean_class=clean_id,
                             pretrained=False,seed_pretrained_stem=False)
    m.load_state_dict(state)
    return m, classes, clean_id

def val_loader(cfg, classes):
    """The run's validation split, by import. This used to re-derive it: load the val h5,
    apply the scheme, align onto the training class order. Same chain the trainer runs, so
    it is now the same code -- see Datasets.build_val_only. Batch size and worker count now
    come from the config instead of being hardcoded 512/8; neither changes what is scored,
    since the loader does not shuffle and does not drop the last batch."""
    split=build_val_only(cfg, verbose=False)
    assert list(split.val.classes)==list(classes), 'val class order != training class order'
    return val_dataloader(cfg, split)

def pclean(model, loader, clean_id, dev, tta):
    model.eval().to(dev); ps=[]; ys=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(dev)
            if tta:
                acc=None
                for v in views(x):
                    p=torch.softmax(model(v),1)
                    acc=p if acc is None else acc+p
                p=acc/5.0
            else:
                p=torch.softmax(model(x),1)
            ps.append(p[:,clean_id].cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)

def miss_at(pcl, y, clean_id, fa):
    """Shared KPI (Metrics.miss_at_fa_sweep). NOTE this script uses the CONSTRAINED-
    MAXIMUM estimator -- best detection subject to false_alarm <= fa -- not the (1-fa)
    quantile the other tools use. Kept as-is so this conversion is a no-op; the two agree
    on smooth score distributions and diverge on ties. See Metrics.py."""
    mass=1.0-pcl; is_clean=(y==clean_id)
    return miss_at_fa_sweep(mass, is_clean, ~is_clean, fa)

def main():
    cfg=load_hyperparameters(CFG)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # states
    s42=torch.load(CKPTS['42'],weights_only=False,map_location='cpu')['state_dict']
    soup={k: (s42[k].float()
              + torch.load(CKPTS['1'],weights_only=False,map_location='cpu')['state_dict'][k].float()
              + torch.load(CKPTS['2'],weights_only=False,map_location='cpu')['state_dict'][k].float())/3.0
          for k in s42}
    _,classes,clean_id=build_model(cfg,s42)
    loader=val_loader(cfg,classes)
    print(f"{'mode':14}{'miss@FA5':>10}{'miss@FA10':>11}")
    for tag,state,tta in [('baseline',s42,False),('TTA(5x)',s42,True),
                          ('soup',soup,False),('soup+TTA',soup,True)]:
        m,_,_=build_model(cfg,state)
        pcl,y=pclean(m,loader,clean_id,dev,tta)
        print(f"{tag:14}{miss_at(pcl,y,clean_id,0.05):10.2f}{miss_at(pcl,y,clean_id,0.10):11.2f}")
    print("(single-seed baselines: s42=10.28 s1=10.19 s2=10.44)")

if __name__=='__main__':
    main()
