"""Paired conditional gradient-variance diagnostic, with equal forward counts.

For fixed clean points and lambda, compare two independent corruptions against
one antithetic pair. This measures noise variance, not variation over training
data or model seeds. Native posterior-bias loss units are unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from models.training import load_checkpoint
from experiments.lambda_repair_eval import CanonicalPosterior


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    m=json.loads((a.run/'manifest.json').read_text());result=load_checkpoint(a.run/'model.pt',device='cuda')
    folder=a.data/m['dataset'];ids=np.load(folder/'fit_indices.npy')[:32]
    raw=np.load(folder/f"train_{m['representation']}.npy",mmap_mode='r')[ids]
    clean=((torch.tensor(raw.copy())-result.normalization_mean)/result.normalization_scale).cuda()
    clean=torch.cat((clean,clean),0);result.model.train()
    parameters=list(result.model.parameters());rows=[]
    for scale in [1/256,1/16,.5,16.]:
        for mode in ['iid','antithetic']:
            field=CanonicalPosterior(result,scale);total=None;norms=[]
            for j in range(32):
                gen=torch.Generator(device='cuda').manual_seed(901+j)
                noise=torch.randn(clean.shape,device='cuda',generator=gen)
                if mode=='antithetic':noise=torch.cat((noise[:32],-noise[:32]),0)
                prediction=field(clean+scale*noise)
                loss=((prediction-clean)/scale).square().mean()
                gradients=torch.autograd.grad(loss,parameters)
                flat=torch.cat([g.flatten() for g in gradients]).double()
                total=flat.clone() if total is None else total+flat
                norms.append(float(flat.square().sum()))
            mean=total/32;mean_norm=float(mean.square().sum())
            variance=(sum(norms)-32*mean_norm)/31
            rows.append(dict(lambda_value=scale,mode=mode,repetitions=32,clean_points=32,forward_examples_per_repeat=64,
                conditional_gradient_variance=variance,squared_mean_gradient=mean_norm,gradient_squared_norms=norms))
            print(json.dumps({k:v for k,v in rows[-1].items() if k!='gradient_squared_norms'}),flush=True)
    for i in range(0,len(rows),2):rows[i+1]['variance_reduction_factor']=rows[i]['conditional_gradient_variance']/rows[i+1]['conditional_gradient_variance']
    dest=a.run/'gradient_variance';dest.mkdir(exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=result.checkpoint_sha256,
        scope='conditional on 32 fixed optimizer-fit observations; equal64 network evaluations',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
