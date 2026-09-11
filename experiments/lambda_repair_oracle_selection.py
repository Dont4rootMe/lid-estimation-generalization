"""Run the unchanged selector on an exact population posterior, with/without trace noise."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from models.training import load_checkpoint
from models.neural_fields import rademacher_probes_like
from experiments.lambda_repair_eval import SCALES,selected
from experiments.lambda_repair_oracle import renderer,reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    m=json.loads((a.run/'manifest.json').read_text());kind=m['dataset'];rep=m['representation']
    result=load_checkpoint(a.run/'model.pt',device='cpu');folder=a.data/kind
    active={'e6_exp_pca':3,'e1_spiral_pca':2,'e8_spaghetti_pca':20,'e8_sphere4_pca':6}[kind]
    d={'e6_exp_pca':2,'e1_spiral_pca':1,'e8_spaghetti_pca':1,'e8_sphere4_pca':5}[kind]
    ids=np.load(folder/'holdout_indices.npy');coeff=np.load(folder/'train_coefficients.npy',mmap_mode='r')[ids,:active].astype(float)
    raw=np.load(folder/f'train_{rep}.npy',mmap_mode='r')[ids]
    normalized=((torch.tensor(raw.copy())-result.normalization_mean)/result.normalization_scale).numpy().astype(float)
    matrix,offset,check=renderer(folder,rep,active);gram=matrix@matrix.T
    assert check['max_gram_difference_from_identity']<1e-6,check
    projections={}
    for count in [16,64]:
        generator=torch.Generator(device='cuda').manual_seed(0);parts=[]
        for start in range(0,len(ids),512):
            x=torch.zeros(min(512,len(ids)-start),raw.shape[1],device='cuda')
            probes=rademacher_probes_like(x,num_probes=count,generator=generator)
            parts.append((probes.double()@torch.tensor(matrix.T,device='cuda')).cpu().numpy())
        projections[count]=np.concatenate(parts)
    saved={};errors=[]
    for scale in SCALES:
        sigma=scale*result.normalization_scale
        values=[reference(kind,q,sigma) for q in coeff]
        mean=np.stack([v[0] for v in values]);jac=np.stack([v[3] for v in values])/sigma**2
        r=np.einsum('nij,ji->n',jac,gram)
        q=(mean@matrix+offset-result.normalization_mean.numpy())/result.normalization_scale
        c=np.square(q-normalized).sum(1)/scale**2
        row=dict(response_exact=r,correction=c,full_exact=r+c,posterior_coefficients=mean)
        for count,probes in projections.items():
            draws=np.einsum('npi,nij,npj->np',probes,jac,probes)
            row[f'full_p{count}']=draws.mean(1)+c
            row[f'response_probes_{count}']=draws
        for key,value in row.items():saved.setdefault(key,[]).append(value)
        high=[reference(kind,q,sigma,high=True) for q in coeff[:8]]
        error=max(max(abs(high[i][1]-values[i][1]),abs(high[i][2]-values[i][2])) for i in range(len(high)))
        errors.append(error)
        print(json.dumps(dict(lambda_value=float(scale),mae_exact=float(abs(r+c-d).mean()),
            mae_p16=float(abs(row['full_p16']-d).mean()),mae_p64=float(abs(row['full_p64']-d).mean()),
            quadrature_check=error)),flush=True)
    output={key:np.stack(value,1) for key,value in saved.items()};target=np.full(len(ids),d)
    summary=[]
    for key in ['full_exact','full_p16','full_p64']:
        curve=output[key];scale,col,diag=selected(curve,target)
        rng=np.random.default_rng(921);boot=[]
        for _ in range(500):
            ix=rng.integers(len(ids),size=len(ids));boot.append(selected(curve[ix],target[ix])[0])
        vals,counts=np.unique(boot,return_counts=True)
        summary.append(dict(estimator=key,selected_lambda=scale,holdout_mae=float(abs(curve[:,col]-d).mean()),
            status=diag['status'],bootstrap_probability_ge16=float(np.mean(np.array(boot)>=16-1e-8)),
            bootstrap=[dict(lambda_value=float(v),n=int(n)) for v,n in zip(vals,counts)]))
    dest=a.run/'oracle_selection';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'arrays.npz',scales=SCALES,query_ids=ids,target=target,**output)
    (dest/'summary.json').write_text(json.dumps(dict(dataset=kind,representation=rep,query_count=len(ids),
        posterior='exact continuous population law; no trained field',trace_batch_size=512,trace_seed=0,
        original_checkpoint_used_only_for_normalization=result.checkpoint_sha256,renderer=check,
        max_quadrature_refinement_error=max(errors),results=summary),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
