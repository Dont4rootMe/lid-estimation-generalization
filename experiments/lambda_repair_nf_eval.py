"""Source OLS5 likelihood readout; holdout selection precedes test evaluation."""
import yaml
import argparse,json
from pathlib import Path
import numpy as np
import torch
from datasets.registry import load_registry
from models.training import load_checkpoint,predict_lid
from experiments.global_campaign_v2 import _prediction_curve
from experiments.lambda_repair_eval import SCALES,selected,CONTRACTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--readout',choices=['ols5','fixed_likelihood'],default='ols5')
    p.add_argument('--selector',choices=['bounded_v2','full_support_v1'],default='bounded_v2')
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    assert (a.run/'complete.json').exists()
    manifest=json.loads((a.run/'manifest.json').read_text());result=load_checkpoint(a.run/'model.pt',device='cuda')
    contracts=yaml.safe_load(CONTRACTS.read_text())['model_contracts']
    spec=next(c['model'] for c in contracts if c['variant_id']=='scale_conditioned_nf')
    registry=load_registry(Path(__file__).resolve().parents[1]/'configs/datasets/registry/paper_benchmarks.yaml',validate_official_coverage=False)
    d=float(registry[manifest['dataset']].expected_lid);folder=a.data/manifest['dataset'];rep=manifest['representation']
    ids=np.load(folder/'holdout_indices.npy');raw=np.load(folder/f'train_{rep}.npy',mmap_mode='r')[ids]
    result.model.double() if a.dtype=='float64' else result.model.float()
    def evaluate(raw,scales):
        return _prediction_curve(predict_lid,result,raw,np.asarray(scales),model=spec,seed=0,batch_size=512,readout=a.readout)
    curve=evaluate(raw,SCALES);target=np.full(len(raw),d)
    scale,col,diag=selected(curve,target,a.selector)
    suffix=('_exact' if a.readout=='fixed_likelihood' else '')+('_'+a.selector if a.selector!='bounded_v2' else '')
    dest=a.run/f'evaluation_nf_{a.dtype}{suffix}';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'holdout.npz',scales=SCALES,full=curve,target=target)
    rng=np.random.default_rng(921);boot=[]
    for _ in range(500):
        idx=rng.integers(len(raw),size=len(raw));boot.append(selected(curve[idx],target[idx],a.selector)[0])
    vals,counts=np.unique(boot,return_counts=True)
    receipt=dict(variant=manifest['variant'],dataset=manifest['dataset'],representation=rep,arm=manifest['arm'],
        checkpoint_sha256=result.checkpoint_sha256,selected_lambda=scale,target_lid=d,status=diag['status'],
        readout='source fixed-likelihood OLS5' if a.readout=='ols5' else 'exact autograd log-scale derivative',
        selector=a.selector,total_steps=result.config.steps,native_best_step=result.best_epoch,
        condition_max_frequency=result.config.max_condition_frequency,
        terminal_decay_steps=result.config.terminal_decay_steps,ema_decay=result.config.ema_decay,
        holdout_mae=float(abs(curve[:,col]-d).mean()),
        selection_bootstrap=[dict(lambda_value=float(v),n=int(n)) for v,n in zip(vals,counts)],
        bootstrap_probability_ge16=float(np.mean(np.array(boot)>=16-1e-8)),
        target_policy='paper_benchmarks_registry_effective_lid',evaluation_dtype=a.dtype,batch_size=512)
    (dest/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    rawtest=np.load(folder/f'test_{rep}.npy');test=evaluate(rawtest,[scale])
    np.savez_compressed(dest/'test_selected.npz',scales=np.array([scale]),full=test,target=np.full(len(test),d))
    receipt['test_mae']=float(abs(test[:,0]-d).mean())
    (dest/'test_metrics.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
