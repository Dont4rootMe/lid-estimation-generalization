"""Paired native flat-data check of a fully learned posterior output.

The unrestricted network starts at zero output instead of copying the noisy
input through an analytic isotropic skip. No basis, rank or covariance enters
the model. Reuse the exact data/noise draw sequence of the retained controls.
"""
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from experiments.fair_protocol import build_model,geometry,resolve,source_identity
from models.non_gaussian_fields import loss
from flat_scaling_control import evaluate

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_gaussian_ambient_20260911'
OUT=BASE/'flat_no_skip_control'


class NoSkipFixture(nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core
        self.coefficient='inverse_total_no_input_skip'
        self._lid_noise_pairing='antithetic_v1'

    def forward(self,y,scale):
        scale=torch.as_tensor(scale,device=y.device,dtype=y.dtype)
        if scale.ndim==0:scale=scale.expand(len(y))
        inverse=torch.rsqrt(1+scale.square())[:,None]
        residual=self.core(y*inverse,scale.log(),(scale[:,None]*inverse).expand_as(y))
        return inverse*residual


def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    cfg,_=resolve('t_flowmatching',geometry('flat_fixture','coefficients',[30]),device='cuda',steps=4000,preflight=True)
    task_rng=torch.Generator(device='cuda').manual_seed(28517)
    rotation,_=torch.linalg.qr(torch.randn((30,30),generator=task_rng,device='cuda'))
    basis,normal=rotation[:,:2],rotation[:,2:]
    torch.manual_seed(3981)
    model=NoSkipFixture(build_model('t_flowmatching',cfg,30).core.cuda()).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-6)
    objective_rng=torch.Generator(device='cuda').manual_seed(15813)
    source=Path(__file__).read_bytes();(OUT/'source.py').write_bytes(source)
    base_result=BASE/'flat_scaling_control/result.json'
    assert json.loads(base_result.read_text())['status']=='complete'
    manifest=dict(status='in_progress',steps=4000,data_seed=28517,model_seed=3981,batch_size=256,
        native_family='student_t_flow',df=5,ambient=30,clean_dimension=2,
        paired_baseline_sha256=hashlib.sha256(base_result.read_bytes()).hexdigest(),
        actual_parameters=sum(p.numel() for p in model.parameters()),known_span_in_model=False,
        initialization='same zero residual core, no analytic input copy',
        common_source=source_identity(),source_sha256=hashlib.sha256(source).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.time();rows=[]
    for step in range(1,4001):
        clean=torch.randn((256,2),generator=task_rng,device='cuda')@basis.T*(15**.5)
        optimizer.zero_grad(set_to_none=True)
        objective=loss('student_t_flow',model,clean,cfg,objective_rng)
        assert torch.isfinite(objective)
        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%500==0:print(json.dumps(dict(step=step,loss=float(objective.detach()),seconds=time.time()-started)),flush=True)
        if step in (1000,4000):rows.extend(evaluate(model,basis,normal,step))
    torch.save(dict(state_dict=model.cpu().state_dict(),manifest=manifest),OUT/'model.pt')
    assert source_identity()==manifest['common_source']
    (OUT/'result.json').write_text(json.dumps(dict(status='complete',seconds=time.time()-started,rows=rows),indent=2)+'\n')
    print(json.dumps(dict(status='complete',rows=[r for r in rows if r['step']==4000])),flush=True)


if __name__=='__main__':main()
