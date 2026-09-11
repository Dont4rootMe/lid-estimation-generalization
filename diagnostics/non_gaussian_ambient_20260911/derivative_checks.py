"""Independent full-coordinate finite differences on actual native checkpoints."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from models import training
from models.neural_fields import exact_divergence


def check(result,normalized,scales,reference32=None):
    assert result.family in {'student_t_flow','pfgmpp'}
    assert result.config.field_projection_rank is None
    model=result.model
    assert not hasattr(model,'basis')
    parameter=next(model.parameters());original_dtype=parameter.dtype
    x=normalized.to(device=parameter.device,dtype=torch.float64)
    records=[];model.double()
    try:
        for scale in scales:
            started=time.time();scale=float(scale)
            response=exact_divergence(model,x,scale).detach()
            estimates=[]
            for step in (1e-5,3e-6):
                finite=torch.zeros(len(x),device=x.device,dtype=x.dtype)
                with torch.no_grad():
                    for j in range(x.shape[1]):
                        direction=torch.zeros_like(x);direction[:,j]=1
                        finite+=(model(x+step*direction,scale)[:,j]-model(x-step*direction,scale)[:,j])/(2*step)
                estimates.append(finite)
            gap=float((estimates[-1]-response).abs().max())
            refinement=float((estimates[0]-estimates[1]).abs().max())
            assert gap<2e-3 and refinement<2e-3,(scale,gap,refinement)
            row=dict(lambda_value=scale,coordinates=x.shape[1],n=len(x),
                finite_difference_gap=gap,step_refinement_gap=refinement,
                exact_trace=response.cpu().numpy().tolist(),seconds=time.time()-started)
            if reference32 is not None:
                precision=float(abs(response.cpu().numpy()-reference32[scale]).max())
                row['float32_vs_float64_trace_gap']=precision
                row['float32_precision_gate']=precision<2e-3
            records.append(row)
    finally:
        model.to(dtype=original_dtype)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result=training.load_checkpoint(args.checkpoint,device='cuda')
    generator=torch.Generator(device='cuda').manual_seed(663)
    x=torch.randn((2,result.model.config.ambient_dim),device='cuda',generator=generator)
    before=exact_divergence(result.model,x,1.).detach().cpu().numpy()
    rows=check(result,x,[1.],{1.:before})
    assert all(r['float32_precision_gate'] for r in rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(status='passed',rows=rows),indent=2)+'\n')
    print(json.dumps(dict(status='passed',rows=rows)),flush=True)


if __name__=='__main__':main()
