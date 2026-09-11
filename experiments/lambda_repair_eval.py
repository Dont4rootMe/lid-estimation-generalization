"""Frozen-checkpoint R/C curves and actual-v2-selector bootstrap.

Default is source-train holdout only. Test predictions require --test and are
computed only after a selection receipt has been written.
"""
import yaml
import argparse
import csv
import json
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from models.training import load_checkpoint, predict_lid
from models.affine_flow import affine_schedule_state, velocity_to_posterior
from models.vp_baseline import VPSchedule
from models.neural_fields import rademacher_probes_like
from experiments.global_campaign_v2 import select_supervised_bounded, initial_supervised_lambdas, _prediction_curve
from datasets.registry import load_registry

CONTRACTS=Path(__file__).resolve().parents[1]/'configs/lambda_repair/native_contracts.yaml'
SCALES=2.**(np.arange(-16,13)/2)


class CanonicalPosterior(nn.Module):
    def __init__(self,result,scale):
        super().__init__();self.model=result.model;self.cfg=result.config;self.family=result.family;self.scale=float(scale)

    def forward(self,x):
        cfg,family,s=self.cfg,self.family,self.scale
        if family=='vp_diffusion':
            schedule=VPSchedule(cfg.vp_beta_min,cfg.vp_beta_max)
            condition=x.new_full((len(x),),schedule.time_for_lambda(s));alpha,_=schedule.coefficients(condition)
            return x+s*self.model(alpha[:,None]*x,condition)
        if family=='rectified_flow':
            alpha=1/(1+s);condition=x.new_full((len(x),),alpha)
            return alpha*x+(1-alpha)*self.model(alpha*x,condition)
        if family=='gaussian_diffusion':return self.model(x,x.new_full((len(x),),s))
        if family=='brownian_schrodinger_bridge':return self.model(x,x.new_full((len(x),),s*s/cfg.bridge_diffusivity))
        condition=x.new_full((len(x),),np.log(s));state=affine_schedule_state(x.new_full((len(x),),s),cfg.flow_schedule)
        native=state.alpha[:,None]*x;prediction=self.model(native,condition)
        if cfg.flow_parameterization=='posterior_mean':return prediction
        return velocity_to_posterior(prediction,native,state)


def curves(result,raw,scales,probes=16,batch_size=128):
    device=next(result.model.parameters()).device;dtype=next(result.model.parameters()).dtype
    x=(torch.as_tensor(np.asarray(raw).copy(),dtype=torch.float32)-result.normalization_mean)/result.normalization_scale
    rows=[]
    result.model.eval()
    for scale in scales:
        field=CanonicalPosterior(result,scale)
        generator=torch.Generator(device=device).manual_seed(0)
        r,c,rp=[],[],[]
        for start in range(0,len(x),batch_size):
            q=x[start:start+batch_size].to(device=device,dtype=dtype).requires_grad_(True)
            if probes==0:
                from models.preconditioned_field import covariance_span_value_and_trace
                value,response=covariance_span_value_and_trace(result.model,q,float(scale))
                r.append(response.cpu().numpy());rp.append(np.empty((len(q),0)))
                c.append(((value.double()-q.detach().double()).square().sum(1)/(scale*scale)).cpu().numpy())
                continue
            value=field(q)
            tape=rademacher_probes_like(q,num_probes=probes,seed=None,generator=generator)
            estimates=[]
            for j in range(probes):
                grad=torch.autograd.grad(value,q,grad_outputs=tape[:,j],retain_graph=j+1<probes)[0]
                estimates.append((grad*tape[:,j]).sum(1).detach())
            samples=torch.stack(estimates,1)
            r.append(samples.mean(1).cpu().numpy());rp.append(samples.cpu().numpy())
            c.append(((value.detach()-q.detach()).square().sum(1)/(scale*scale)).cpu().numpy())
        rows.append((np.concatenate(r),np.concatenate(c),np.concatenate(rp)))
    response=np.stack([v[0] for v in rows],1).astype(float)
    correction=np.stack([v[1] for v in rows],1).astype(float)
    return dict(scales=np.array(scales),response=response,correction=correction,full=response+correction,
                response_probes=np.stack([v[2] for v in rows],1))


def selected(curve,target,selector='bounded_v2'):
    if selector=='full_support_v1':
        errors=np.abs(curve-np.asarray(target)[:,None]).mean(0)
        col=int(np.flatnonzero(errors<=errors.min()+1e-12)[0])
        return float(SCALES[col]),col,dict(status='scale_unresolved' if col in [0,len(SCALES)-1] else 'selected',
            protocol='held_out_source_train_supervised_mae_full_support_v1')
    if selector!='bounded_v2':raise ValueError('unknown scale selector')
    initial=initial_supervised_lambdas()
    def values(grid):return curve[:,[int(np.argmin(abs(SCALES-s))) for s in grid]]
    grid,pred,index,diag=select_supervised_bounded(initial,values(initial),target,evaluate=values)
    scale=float(grid[index]);col=int(np.argmin(abs(SCALES-scale)))
    return scale,col,diag


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--test',action='store_true')
    parser.add_argument('--probes',type=int,default=16)
    parser.add_argument('--selector',choices=['bounded_v2','full_support_v1'],default='bounded_v2')
    parser.add_argument('--batch-size',type=int,default=128);args=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();manifest=json.loads((args.run/'manifest.json').read_text())
    if not (args.run/'complete.json').is_file() and not (args.run/'interim.json').is_file():
        raise RuntimeError('Evaluation requires completed training and its portable native-best checkpoint')
    result=load_checkpoint(args.run/'model.pt',device='cuda')
    folder=args.data/manifest['dataset'];rep=manifest['representation']
    indices=np.load(folder/'holdout_indices.npy')
    raw=np.load(folder/f'train_{rep}.npy',mmap_mode='r')[indices]
    registry=load_registry(Path(__file__).resolve().parents[1]/'configs/datasets/registry/paper_benchmarks.yaml',validate_official_coverage=False)
    # The upstream registry corrects legacy labels on Exp/Crescent/Spaghetti.
    # Use exactly those effective campaign targets, not raw archived lid.npy.
    true_lid=float(registry[manifest['dataset']].expected_lid)
    target=np.full(len(indices),true_lid)
    name='evaluation' if (args.batch_size,args.probes)==(128,16) else f'evaluation_b{args.batch_size}_p{args.probes}'
    if args.selector!='bounded_v2':name+='_'+args.selector
    dest=args.run/name;dest.mkdir(exist_ok=True)
    cache_identity=dict(checkpoint_sha256=result.checkpoint_sha256,probes=args.probes,batch_size=args.batch_size,
        target_lid=true_lid,query_count=len(indices))
    if (dest/'holdout.npz').exists():
        with np.load(dest/'holdout.npz') as z:values={k:z[k] for k in z.files}
        np.testing.assert_array_equal(values['target'],target)
        assert values['response_probes'].shape[2]==args.probes
        if (dest/'cache_identity.json').exists():
            assert json.loads((dest/'cache_identity.json').read_text())==cache_identity
    else:
        values=curves(result,raw,SCALES,args.probes,args.batch_size);values['target']=target
        np.savez_compressed(dest/'holdout.npz',**values)
    (dest/'cache_identity.json').write_text(json.dumps(cache_identity,indent=2)+'\n')
    scale,col,diag=selected(values['full'],target,args.selector)
    rng=np.random.default_rng(921);boot=[]
    for _ in range(500):
        idx=rng.integers(len(target),size=len(target));boot.append(selected(values['full'][idx],target[idx],args.selector)[0])
    unique,counts=np.unique(np.round(boot,10),return_counts=True)
    receipt=dict(variant=manifest['variant'],dataset=manifest['dataset'],representation=rep,arm=manifest['arm'],
        training_status='interim' if manifest.get('interim') else 'complete',
        batch_size=args.batch_size,field_backbone=result.config.field_backbone,
        noise_pairing=result.config.noise_pairing,total_steps=result.config.steps,
        field_residual_scaling=result.config.field_residual_scaling,
        field_residual_width=result.config.field_residual_width,
        training_target=result.config.training_target,training_target_start_step=result.config.training_target_start_step,
        selector=args.selector,
        terminal_decay_steps=result.config.terminal_decay_steps,ema_decay=result.config.ema_decay,
        checkpoint_sha256=result.checkpoint_sha256,target_lid=true_lid,target_policy='paper_benchmarks_registry_effective_lid',selected_lambda=scale,status=diag['status'],
        holdout_mae=float(np.abs(values['full'][:,col]-target).mean()),mean_response=float(values['response'][:,col].mean()),
        mean_correction=float(values['correction'][:,col].mean()),native_best_step=result.best_epoch,
        selection_bootstrap=[dict(lambda_value=float(s),n=int(n)) for s,n in zip(unique,counts)],
        bootstrap_probability_ge16=float(np.mean(np.array(boot)>=16-1e-8)),probes=args.probes,
        diagnosis=('query bootstrap conditional on fixed trained weights and exact trace' if args.probes==0
                   else 'query bootstrap conditional on fixed trained weights and trace draws'))
    (dest/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Verify the canonical formula against source interfaces using exact traces
    # on two queries at one non-singular scale (the 784-coordinate audit is explicit).
    if not (dest/'source_parity.json').exists():
        contracts=yaml.safe_load(CONTRACTS.read_text())['model_contracts']
        spec=next(c['model'] for c in contracts if c['variant_id']==manifest['variant']).copy()
        spec['derivative_backend']='active_exact' if args.probes==0 else 'hutchinson';spec['trace_probes']=args.probes
        actual_sample=_prediction_curve(predict_lid,result,raw[:args.batch_size],np.array([scale]),model=spec,seed=0,
            batch_size=args.batch_size,readout='full')[:,0]
        sample_difference=float(np.max(abs(actual_sample-values['full'][:args.batch_size,col])))
        np.testing.assert_allclose(actual_sample,values['full'][:args.batch_size,col],rtol=2e-5,atol=5e-4)
        spec['derivative_backend']='exact';spec['trace_probes']=0
        # Low-dimensional exact check is supplemented by the full D Gaussian unit tests.
        check_scale=2.;x=((torch.as_tensor(raw[:2].copy())-result.normalization_mean)/result.normalization_scale).double().cuda()
        result.model.double();field=CanonicalPosterior(result,check_scale)
        expected=[]
        for q in x:
            jac=torch.func.jacrev(lambda y:field(y[None])[0])(q)
            with torch.no_grad():correction=(field(q[None])[0]-q).square().sum()/(check_scale**2)
            expected.append(float(jac.trace()+correction))
        actual=_prediction_curve(predict_lid,result,raw[:2],np.array([check_scale]),model=spec,seed=0,batch_size=2,readout='full')[:,0]
        discrepancy=float(np.max(abs(actual-expected)));assert discrepancy<1e-7,(actual,expected)
        (dest/'source_parity.json').write_text(json.dumps({'n':2,'lambda':2.,'ambient_dim':len(x[0]),
            'float64_exact_max_difference':discrepancy,'source_stochastic_batch_size':args.batch_size,
            'source_stochastic_n':len(actual_sample),'source_stochastic_max_difference':sample_difference})+'\n')
        result.model.float()
    if args.test:
        rawtest=np.load(folder/f'test_{rep}.npy');testtarget=np.full(len(rawtest),true_lid)
        test=curves(result,rawtest,[scale],args.probes,args.batch_size);test['target']=testtarget
        np.savez_compressed(dest/'test_selected.npz',**test)
        receipt['test_mae']=float(abs(test['full'][:,0]-testtarget).mean())
        (dest/'test_metrics.json').write_text(json.dumps(receipt,indent=2)+'\n')
    receipt['duration_seconds']=time.time()-start
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
