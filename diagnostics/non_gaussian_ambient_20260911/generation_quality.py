"""Fixed-checkpoint native ODE samples; no model, scale or data tuning.

For both native interfaces, canonical coordinates obey dy/dlog(lambda)=y-b.
No projection or denoising teacher is used in this sampler. Known generator
coordinates are used only to measure normal error and to draw diagnostic plots.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from datasets.registry import load_split
from experiments.fair_campaign import verify_measurements,inventory,dataset_spec,write_json,file_sha
from experiments import global_campaign as partition_tools
from models import training
from models.non_gaussian_fields import student_unit_rms

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_gaussian_ambient_20260911'
SOURCE=Path(__file__).read_bytes()


@torch.no_grad()
def sample(model,initial,steps):
    times=np.linspace(math.log(64.),math.log(1/256),steps+1)
    y=initial.clone()
    for left,right in zip(times[:-1],times[1:]):
        h=float(right-left)
        first=y-model(y,math.exp(left))
        predictor=y+h*first
        second=predictor-model(predictor,math.exp(right))
        y=y+.5*h*(first+second)
        if not torch.isfinite(y).all():raise FloatingPointError('native Heun trajectory became nonfinite')
    return model(y,1/256),y


def squared_distances(a,b):
    return (a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*a@b.T).clamp_min(0)


def metrics(generated,test,fit,renderer,offset,mean,rms):
    # All distances below retain every supplied ambient coordinate.
    g=torch.as_tensor(generated,dtype=torch.float64,device='cuda')
    t=torch.as_tensor(test,dtype=torch.float64,device='cuda')
    f=torch.as_tensor(fit,dtype=torch.float64,device='cuda')
    data_total_variance=float(t.var(0,unbiased=True).sum())
    cross=squared_distances(g,t).sqrt()
    gg=squared_distances(g,g).sqrt();tt=squared_distances(t,t).sqrt()
    gg.fill_diagonal_(0);tt.fill_diagonal_(0)
    energy=2*cross.mean()-gg.sum()/(len(g)*(len(g)-1))-tt.sum()/(len(t)*(len(t)-1))
    gf=squared_distances(g,f).amin(1).sqrt();tf=squared_distances(t,f).amin(1).sqrt()
    matrix=torch.as_tensor(renderer,dtype=torch.float64,device='cuda')
    origin=torch.as_tensor(offset,dtype=torch.float64,device='cuda')
    coordinate=(g-origin)@matrix.T@torch.linalg.inv(matrix@matrix.T)
    normal=g-(coordinate@matrix+origin)
    true_coord=(t-origin)@matrix.T@torch.linalg.inv(matrix@matrix.T)
    true_normal=t-(true_coord@matrix+origin)
    return dict(generated_n=len(g),test_n=len(t),fit_neighbor_bank_n=len(f),
        total_variance_ratio=float(g.var(0,unbiased=True).sum())/data_total_variance,
        mean_shift_over_data_rms_norm=float((g.mean(0)-t.mean(0)).square().sum().sqrt())/math.sqrt(data_total_variance),
        energy_distance_over_data_rms_norm=float(energy)/math.sqrt(data_total_variance),
        generated_to_fit_distance_over_data_rms_norm=float(gf.mean())/math.sqrt(data_total_variance),
        test_to_fit_distance_over_data_rms_norm=float(tf.mean())/math.sqrt(data_total_variance),
        generated_normal_distance_over_data_rms_norm=float(normal.square().sum(1).mean().sqrt())/math.sqrt(data_total_variance),
        test_normal_distance_over_data_rms_norm=float(true_normal.square().sum(1).mean().sqrt())/math.sqrt(data_total_variance)),coordinate.cpu().numpy(),true_coord.cpu().numpy()


def evaluate(run):
    run=Path(run);done=verify_measurements(run/'complete.json');dest=run/'generation_quality'
    assert done['kind']=='benchmark' and done['steps_completed']==128000 and done['test_n']==1000
    source_hash=hashlib.sha256(SOURCE).hexdigest()
    if (dest/'summary.json').exists():
        saved=json.loads((dest/'summary.json').read_text())
        assert saved['checkpoint_sha256']==done['checkpoint_sha256'] and saved['diagnostic_sha256']==source_hash
        assert saved['samples_sha256']==file_sha(dest/'samples.npz')
        return
    if dest.exists():dest.rename(run/f'generation_partial_{time.time_ns()}')
    dest.mkdir();(dest/'source.py').write_bytes(SOURCE)
    started=time.time();result=training.load_checkpoint(run/'model.pt',device='cuda')
    result.model.eval()
    assert result.family in {'student_t_flow','pfgmpp'}
    assert result.config.field_projection_rank is None and not hasattr(result.model,'basis')
    _,cells=inventory();cell=next(c for c in cells if c.key==done['cell_key'])
    spec=dataset_spec(cell);data=BASE/'data/benchmarks'
    test=load_split(data,spec,'test',representation=cell.representation,mmap_mode='r')
    train=load_split(data,spec,'train',representation=cell.representation,mmap_mode='r')
    with np.load(run/'holdout_curve.npz') as z:holdout_ids=z['query_ids']
    full_fit_ids=np.flatnonzero(~np.isin(np.arange(len(train.features)),holdout_ids))
    assert partition_tools._array_sha(full_fit_ids)==done['fit_indices_sha256']
    assert partition_tools._array_sha(holdout_ids)==done['holdout_indices_sha256']
    fit_ids=full_fit_ids[:8192]
    test_raw=np.asarray(test.features).reshape(len(test.features),-1)
    fit_raw=np.asarray(train.features).reshape(len(train.features),-1)[fit_ids]
    pop=BASE/'population'/f'{cell.dataset}__{cell.representation}'
    geometry=json.loads((pop/'geometry.json').read_text())
    assert geometry['normalization_sha256']==result.preprocessing_sha256
    assert geometry['fit_indices_sha256']==done['fit_indices_sha256']
    with np.load(pop/'queries.npz') as z:renderer=z['renderer'];offset=z['offset']
    generator=torch.Generator(device='cuda').manual_seed(10522)
    initial=64*student_unit_rms((512,done['resolved']['geometry']['ambient_dim']),result.config.kernel_df,
        device='cuda',dtype=torch.float32,generator=generator)
    outputs=[];endpoints=[]
    for chunk in initial.split(128):
        output,endpoint=sample(result.model,chunk,512)
        outputs.append(output);endpoints.append(endpoint)
    samples=torch.cat(outputs);endpoints=torch.cat(endpoints)
    refined,_=sample(result.model,initial[:64],1024)
    paired=(refined-samples[:64]).square().mean(1).sqrt()
    # Refinement is recorded, never used to select a model or tune LID lambda.
    # A failed numerical check remains visible rather than silently replacing
    # this predeclared512-step sample set.
    mean=result.normalization_mean.numpy();rms=result.normalization_scale
    raw=samples.cpu().numpy()*rms+mean
    score,coordinates,test_coordinates=metrics(raw,test_raw,fit_raw,renderer,offset,mean,rms)
    record=dict(status='complete',variant=done['variant'],cell_key=done['cell_key'],
        checkpoint_sha256=done['checkpoint_sha256'],diagnostic_sha256=source_hash,
        native_kernel_df=result.config.kernel_df,ode='dy/dlog(lambda)=y-b(y,lambda)',
        initial_law='unit-RMS native radial Student noise at lambda64; finite-start approximation',
        integrator='Heun,512 uniform log-scale steps, final posterior at lambda1/256',
        refinement_n=64,refinement_steps=1024,refinement_rms_per_coordinate=float(paired.square().mean().sqrt()),
        refinement_max_per_coordinate_rms=float(paired.max()),
        refinement_gate_passed=bool(float(paired.square().mean().sqrt())<.005),
        seed=10522,metrics=score,seconds=time.time()-started,
        train_files_sha256=done['train_files_sha256'],test_files_sha256=done['test_files_sha256'],
        fit_indices_sha256=done['fit_indices_sha256'],holdout_indices_sha256=done['holdout_indices_sha256'],
        evaluation_geometry_sha256={name:file_sha(pop/name) for name in ('geometry.json','queries.npz')},
        interpretation='sample diagnostics in the full ambient space; no test-based tuning, no empirical posterior or projection in the native trajectory')
    np.savez_compressed(dest/'samples.npz',raw=raw,normalized=samples.cpu().numpy(),
        end_state=endpoints.cpu().numpy(),initial=initial.cpu().numpy(),refined_first64=refined.cpu().numpy(),
        coordinates=coordinates,test_coordinates=test_coordinates,test_raw=test_raw,
        fit_neighbor_bank_ids=fit_ids,renderer=renderer,offset=offset)
    record['samples_sha256']=file_sha(dest/'samples.npz')
    write_json(dest/'summary.json',record)
    print(json.dumps(dict(event='generation_complete',**record)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.run:return evaluate(args.run)
    if not args.watch:raise ValueError('provide --run or --watch')
    campaign=json.loads((BASE/'full_budget/campaign.json').read_text())
    pending={row['name'] for row in campaign['cells']}
    while pending:
        changed=False
        for name in sorted(pending):
            run=BASE/'full_budget'/name
            if (run/'complete.json').exists():
                evaluate(run);pending.remove(name);changed=True
        if pending and not changed:
            if (BASE/'full_budget/launcher_complete.json').exists():
                receipt=json.loads((BASE/'full_budget/launcher_complete.json').read_text())
                if receipt['failures']:raise ValueError('campaign has failed cells')
            time.sleep(15)
    write_json(BASE/'generation_complete.json',dict(status='complete',cells=8))


if __name__=='__main__':main()
