"""Continuous-law posterior checks on fixed source-train holdout queries.

Oracle laws are evaluation only. A renderer is fitted from optimizer-fit pairs;
the learned network never receives coefficients, geometry or oracle targets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from models.training import load_checkpoint
from experiments.lambda_repair_eval import CanonicalPosterior

from experiments.lambda_repair_reference import (tube_posterior,spiral_posterior,
    spaghetti_posterior,mixture_posterior,benchmark_components,SOURCE_SNAPSHOTS)

SCALES=np.array([1/256,1/64,1/16,.25,1.,2.,4.,8.,16.,32.,64.])


def renderer(folder,representation,active):
    fit=np.load(folder/'fit_indices.npy')[:4096]
    coefficients=np.load(folder/'train_coefficients.npy',mmap_mode='r')[fit,:active].astype(float)
    raw=np.load(folder/f'train_{representation}.npy',mmap_mode='r')[fit].astype(float)
    design=np.column_stack((coefficients,np.ones(len(fit))))
    fitted=np.linalg.lstsq(design,raw,rcond=None)[0]
    residual=raw-design@fitted
    gram=fitted[:-1]@fitted[:-1].T
    return fitted[:-1],fitted[-1],dict(fit_count=len(fit),max_fit_residual=float(abs(residual).max()),
        max_gram_difference_from_identity=float(abs(gram-np.eye(active)).max()))


def reference(kind,query,sigma,high=False):
    if kind=='e6_exp_pca':
        value=tube_posterior(query,sigma,order=256 if high else 128,cutoff=12. if high else 10.)
    elif kind=='e1_spiral_pca':
        value=spiral_posterior(query,sigma,order=96 if high else 48,cutoff=12. if high else 10.)
    elif kind=='e8_spaghetti_pca':
        value=spaghetti_posterior(query,sigma,order=32 if high else 16,cutoff=14. if high else 12.)
    elif kind=='e8_sphere4_pca':
        value=mixture_posterior(query,sigma,*benchmark_components())
        return value['posterior'],value['response'],value['correction'],value['covariance']
    else:raise ValueError(kind)
    return value.mean,value.response,value.correction,value.posterior_covariance


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--count',type=int,default=64)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((a.run/'manifest.json').read_text());kind=manifest['dataset'];rep=manifest['representation']
    result=load_checkpoint(a.run/'model.pt',device='cuda');result.model.double().eval()
    folder=a.data/kind;ids=np.load(folder/'holdout_indices.npy')[:a.count]
    active={'e6_exp_pca':3,'e1_spiral_pca':2,'e8_spaghetti_pca':20,'e8_sphere4_pca':6}[kind]
    raw=np.load(folder/f'train_{rep}.npy',mmap_mode='r')[ids]
    coefficients=np.load(folder/'train_coefficients.npy',mmap_mode='r')[ids,:active].astype(float)
    matrix,offset,render_check=renderer(folder,rep,active)
    render_check['holdout_max_residual']=float(abs(coefficients@matrix+offset-raw).max())
    assert render_check['max_gram_difference_from_identity']<1e-6,render_check
    x=((torch.as_tensor(raw.copy())-result.normalization_mean)/result.normalization_scale).double().cuda()
    basis=torch.as_tensor(np.linalg.qr(matrix.T)[0],dtype=torch.float64,device='cuda')
    start=time.time();rows=[];saved={};rng=torch.Generator(device='cuda').manual_seed(921)
    tape=torch.randint(0,2,(len(x),64,len(x[0])),device='cuda',generator=rng).double()*2-1
    for s in SCALES:
        sigma=s*result.normalization_scale;refs=[reference(kind,q,sigma) for q in coefficients]
        oq=np.stack([z[0] for z in refs]);orr=np.array([z[1] for z in refs])
        oq=(oq@matrix+offset-result.normalization_mean.numpy())/result.normalization_scale
        oc=np.square(oq-x.cpu().numpy()).sum(1)/s**2
        check=[reference(kind,q,sigma,high=True) for q in coefficients[:8]]
        quadrature_error=max(max(abs(check[i][1]-refs[i][1]),abs(check[i][2]-refs[i][2])) for i in range(len(check)))
        field=CanonicalPosterior(result,s);query=x.detach().clone().requires_grad_(True);q=field(query)
        rspan=torch.zeros(len(x),device='cuda',dtype=torch.float64)
        for j in range(active):
            v=basis[:,j].expand_as(query)
            derivative=torch.autograd.grad(q,query,grad_outputs=v,retain_graph=True)[0]
            rspan+=(derivative*v).sum(1).detach()
        trace=[]
        for j in range(tape.shape[1]):
            derivative=torch.autograd.grad(q,query,grad_outputs=tape[:,j],retain_graph=j+1<tape.shape[1])[0]
            trace.append((derivative*tape[:,j]).sum(1).detach())
        trace=torch.stack(trace,1).cpu().numpy()
        # For the covariance model, every posterior derivative lies in this
        # span by construction; the trace there is exact, with P64 as a check.
        exact_span=result.config.field_preconditioning=='covariance_span_v1'
        r=rspan.cpu().numpy() if exact_span else trace.mean(1)
        learned=q.detach().cpu().numpy();c=np.square(learned-x.cpu().numpy()).sum(1)/s**2
        error=learned-oq;energy=np.square(error).sum(1)/s**2
        d={'e6_exp_pca':2,'e1_spiral_pca':1,'e8_spaghetti_pca':1,'e8_sphere4_pca':5}[kind]
        row=dict(lambda_value=float(s),raw_sigma=float(sigma),query_count=len(x),target_lid=d,
            learned_lid_mae=float(abs(r+c-d).mean()),oracle_lid_mae=float(abs(orr+oc-d).mean()),
            learned_vs_oracle_full_mae=float(abs(r+c-orr-oc).mean()),
            response_error_mae=float(abs(r-orr).mean()),posterior_error_over_lambda_rms=float(np.sqrt(energy.mean())),
            learned_response_mean=float(r.mean()),learned_correction_mean=float(c.mean()),
            oracle_response_mean=float(orr.mean()),oracle_correction_mean=float(oc.mean()),
            span_response_mean=float(rspan.mean()),ambient_minus_span_trace_p64_estimate=float((trace.mean(1)-rspan.cpu().numpy()).mean()),
            exact_trace=exact_span,quadrature_order_and_cutoff_error=float(quadrature_error))
        rows.append(row)
        for key,value in dict(posterior=learned,response=r,correction=c,oracle_posterior=oq,
                oracle_response=orr,oracle_correction=oc,response_probes=trace,span_response=rspan.cpu().numpy()).items():
            saved.setdefault(key,[]).append(value)
        print(json.dumps(row),flush=True)
    dest=a.run/'continuous_oracle';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'arrays.npz',scales=SCALES,query_ids=ids,**{k:np.stack(v) for k,v in saved.items()})
    (dest/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (dest/'manifest.json').write_text(json.dumps(dict(run_checkpoint=result.checkpoint_sha256,
        query_partition='first fixed source-train holdout indices',test_queries=False,renderer=render_check,
        normalization_rms=result.normalization_scale,seconds=time.time()-start,
        reference_source_sha256=hashlib.sha256(Path(__file__).with_name('lambda_repair_reference.py').read_bytes()).hexdigest(),
        original_reference_sources=SOURCE_SNAPSHOTS),indent=2)+'\n')


if __name__=='__main__':main()
