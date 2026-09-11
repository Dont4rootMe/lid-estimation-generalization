"""Run unchanged contracts in this or the untouched producing checkout.

Invoke by absolute path with each checkout as cwd and PYTHONPATH=dependencies:.
It intentionally resolves models from that working directory.
"""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from models import training


def main():
    p=argparse.ArgumentParser();p.add_argument('--contracts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    x=np.random.default_rng(19).normal(size=(96,30)).astype(np.float32)
    contracts=json.loads(a.contracts.read_text())['model_contracts']
    for contract in contracts:
        cfg=training.TrainingConfig.from_mapping(contract['model']['training'])
        cfg=replace(cfg,steps=20,warmup_steps=min(cfg.warmup_steps,5),validation_interval_steps=5,
                    batch_size=32,num_workers=0)
        result=training.train_model(contract['model']['family'],x[:80],x[80:],cfg,
                                    a.output/(contract['variant_id']+'.pt'))
        print(contract['variant_id'],result.metrics,flush=True)


if __name__=='__main__':main()
