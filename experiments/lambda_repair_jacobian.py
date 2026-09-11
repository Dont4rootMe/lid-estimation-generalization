"""Separate the direct coordinate skip from the learned nonlinear Jacobian.

This is accounting on frozen original models, never an inference alteration.
Full float64 ambient Jacobians on two fixed holdout observations avoid probes.
"""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models.training import load_checkpoint
from experiments.lambda_repair_eval import CanonicalPosterior
from experiments.lambda_repair_oracle import renderer


class NativeLinear(nn.Module):
    def __init__(self, matrix):
        super().__init__();self.register_buffer('matrix',matrix.detach())
    def forward(self, inputs, condition):return inputs@self.matrix.T


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    m=json.loads((a.run/'manifest.json').read_text());r=load_checkpoint(a.run/'model.pt',device='cuda')
    if r.config.field_preconditioning is not None:raise ValueError('original model required')
    r.model.double().eval();folder=a.data/m['dataset']
    ids=np.load(folder/'holdout_indices.npy')[:2]
    raw=np.load(folder/f"train_{m['representation']}.npy",mmap_mode='r')[ids]
    x=((torch.tensor(raw.copy())-r.normalization_mean)/r.normalization_scale).double().cuda()
    d=len(x[0]);active={'e6_exp_pca':3,'e8_spaghetti_pca':20,'e1_spiral_pca':2,'e8_sphere4_pca':6}[m['dataset']]
    matrix,_,_=renderer(folder,m['representation'],active)
    u=torch.tensor(np.linalg.qr(matrix.T)[0],dtype=torch.float64,device='cuda')
    linear=r.model.decoder[-1].weight[:,:d]
    skip=replace(r,model=NativeLinear(linear));rows=[];jacobians=[];skip_jacobians=[]
    for scale in [1/16,.5,2.,64.]:
        field=CanonicalPosterior(r,scale);skipfield=CanonicalPosterior(skip,scale)
        for i,q in enumerate(x):
            j=torch.func.jacrev(lambda y:field(y[None])[0])(q)
            js=torch.func.jacrev(lambda y:skipfield(y[None])[0])(q)
            residual=j-js;span=(u.T@j@u).trace()
            rows.append(dict(lambda_value=scale,holdout_index=int(ids[i]),
                posterior_response=float(j.trace()),coordinate_skip_response=float(js.trace()),
                nonlinear_response=float(residual.trace()),span_response=float(span),
                orthogonal_response=float(j.trace()-span),
                nonlinear_jacobian_frobenius=float(residual.norm()),
                asymmetric_jacobian_frobenius=float((j-j.T).norm()),
                negative_symmetric_trace_mass=float((-torch.linalg.eigvalsh((j+j.T)/2)).clamp_min(0).sum())))
            jacobians.append(j.detach().cpu().numpy());skip_jacobians.append(js.cpu().numpy())
    dest=a.run/'jacobian_accounting';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'arrays.npz',jacobian=np.stack(jacobians),coordinate_skip_jacobian=np.stack(skip_jacobians))
    (dest/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=r.checkpoint_sha256,
        query_count=2,ambient_dim=d,rows=rows),indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
