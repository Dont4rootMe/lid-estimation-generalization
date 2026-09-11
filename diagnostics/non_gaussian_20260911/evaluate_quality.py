"""Fixed-checkpoint response, continuous-law and denoising quality checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from datasets.registry import load_split
from experiments.fair_campaign import (verify_measurements,inventory,dataset_spec,
    prediction_spec,write_json,file_sha)
from experiments import global_campaign as v1, global_campaign_v2 as v2
from experiments import fair_measurements as measurement
from experiments.lambda_repair_eval import CanonicalPosterior
from models import training
from models.non_gaussian_fields import student_unit_rms
from models.preconditioned_field import covariance_span_value_and_trace

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_gaussian_20260911'
DATA=BASE/'data/benchmarks'


def interval(values,seed=851,replicates=4000):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(replicates//200):
        ids=rng.integers(len(values),size=(200,len(values)))
        samples.extend(values[ids].mean(1))
    return np.quantile(samples,[.025,.975]).tolist()


def evaluate(run,device='cuda'):
    start=time.time();run=Path(run);done=verify_measurements(run/'complete.json')
    dest=run/'quality'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir()
    (dest/'evaluate_quality.py').write_bytes(Path(__file__).read_bytes())
    cfg,cells=inventory();cell=next(c for c in cells if c.key==done['cell_key'])
    model=training.load_checkpoint(run/'model.pt',device=device)
    geo=done['resolved']['geometry'];spec=dataset_spec(cell)
    loaded=load_split(DATA,spec,'train',representation=cell.representation,mmap_mode='r')
    raw=loaded.features.reshape(len(loaded.features),-1)
    part=v1.partition_source_train(raw,loaded.lid,selection=cfg['campaign']['selection'],seed=0)
    assert v1._array_sha(part.selection_indices)==done['holdout_indices_sha256']
    contract=prediction_spec(done['variant'],geo)
    def curve(query,scales):
        return v2._prediction_curve(training.predict_lid,model,query,np.asarray(scales),
            model=contract,seed=0,batch_size=128,readout='response')
    scales=measurement.common_scales()
    with np.load(run/'holdout_curve.npz') as z:
        targets=z['target'];holdout=z['prediction'] if done['primary_readout']=='response' else None
    if holdout is None:holdout=curve(part.selection_features,scales)
    selection=measurement.supervised_common_selection(holdout,scales,targets)
    receipt=dict(status='frozen',checkpoint_sha256=model.checkpoint_sha256,selection=selection,
        readout='response',track='primary_native_response' if done['primary_readout']=='response'
            else 'predeclared_secondary_response_selection',
        source_train_holdout_only=True,prior_primary_test_artifacts_exist=True,
        secondary_test_inference_started=False,
        note='This rule was fixed before training; the benchmark primary receipt remains unchanged.')
    write_json(dest/'response_selection.json',receipt)
    frozen=file_sha(dest/'response_selection.json')
    test=load_split(DATA,spec,'test',representation=cell.representation,mmap_mode='r')
    test_raw=test.features.reshape(len(test.features),-1)
    target=np.asarray(test.lid).reshape(-1)
    if done['primary_readout']=='response':
        with np.load(run/'test_scale_curves.npz') as z:predictions=z['common_prediction']
    else:predictions=curve(test_raw,scales)
    chosen=predictions[:,selection['selected_index']]
    kneedle,index,fallback=measurement.pointwise(predictions,scales,geo['ambient_dim'])
    np.savez_compressed(dest/'response_curves.npz',scales=scales,
        holdout=holdout,holdout_target=targets,holdout_query_ids=part.selection_indices,
        test=predictions,test_target=target,test_query_ids=np.arange(len(target)),
        kneedle_prediction=kneedle,kneedle_index=index,kneedle_fallback=fallback)
    assert file_sha(dest/'response_selection.json')==frozen
    result=dict(status='in_progress',variant=done['variant'],cell_key=cell.key,
        primary_metrics=done['metrics'],primary_selected_lambda=done['selected_lambda'],
        primary_automatic=done['automatic_metrics'],response_selection=selection,
        response_metrics=dict(measurement.metric(chosen,target),mae_query_bootstrap_ci95=interval(abs(chosen-target))),
        response_automatic=dict(measurement.metric(kneedle,target),fallback_n=int(fallback.sum()),
            boundary_n=int(((index==0)|(index==len(scales)-1)).sum())),
        practical_half_unit_mae_check=bool(np.mean(abs(chosen-target))<.5),
        uncertainty='fixed-checkpoint, fixed-selection query bootstrap; not retraining uncertainty')
    # All curve/pointwise summaries above are frozen postprocessing, not tuning.
    # Continuous-law comparison uses a preselected, separate holdout subset.
    population=BASE/'population'/f'{cell.dataset}__{cell.representation}'
    geometry=json.loads((population/'geometry.json').read_text())
    assert geometry['normalization_sha256']==model.preprocessing_sha256
    assert geometry['fit_indices_sha256']==done['fit_indices_sha256']
    kernel=done['variant'] if model.family in ('student_t_flow','pfgmpp') else 'gaussian'
    with np.load(population/'queries.npz') as z:
        query=z['raw'];ids=z['query_ids'];renderer=z['renderer'];offset=z['offset']
    with np.load(population/(kernel+'.npz')) as z:
        oracle_r=z['response'];oracle_mean=z['mean']
        assert np.array_equal(z['query_ids'],ids) and np.array_equal(z['scales'],scales)
    assert np.array_equal(ids,part.selection_indices[:len(ids)])
    normalized=(torch.tensor(query,dtype=torch.float32)-model.normalization_mean)/model.normalization_scale
    normalized=normalized.to(device)
    posterior=[]
    with torch.no_grad():
        for scale in scales:posterior.append(CanonicalPosterior(model,scale)(normalized).cpu().numpy())
    posterior=np.asarray(posterior)
    oracle_normalized=(oracle_mean@renderer+offset-model.normalization_mean.numpy())/model.normalization_scale
    learned_r=holdout[:len(ids)].T
    energy=np.sum((posterior-oracle_normalized)**2,axis=2)/scales[:,None]**2
    np.savez_compressed(dest/'continuous_comparison.npz',scales=scales,query_ids=ids,
        learned_response=learned_r,oracle_response=oracle_r,learned_posterior=posterior,
        oracle_posterior=oracle_normalized,posterior_error_over_lambda_squared=energy)
    result['continuous_law']=[dict(lambda_value=float(scale),n=len(ids),
        learned_response_mean=float(learned_r[j].mean()),oracle_response_mean=float(oracle_r[j].mean()),
        learned_response_mae=float(abs(learned_r[j]-geometry['true_lid']).mean()),
        oracle_response_mae=float(abs(oracle_r[j]-geometry['true_lid']).mean()),
        learned_vs_oracle_response_mae=float(abs(learned_r[j]-oracle_r[j]).mean()),
        posterior_error_over_lambda_rms=float(np.sqrt(energy[j].mean()))) for j,scale in enumerate(scales)]
    # Fresh own-kernel corruptions of128 test objects,8 per object. The zero-head
    # baseline is exactly this architecture's initial covariance preconditioner.
    clean=(torch.tensor(np.asarray(test_raw[:128]).copy(),dtype=torch.float32)-model.normalization_mean)/model.normalization_scale
    clean=clean.repeat(8,1).to(device)
    generator=torch.Generator(device=device).manual_seed(488)
    noise=(student_unit_rms(clean.shape,model.config.kernel_df,device=device,
        dtype=clean.dtype,generator=generator) if model.family in ('student_t_flow','pfgmpp') else
        torch.randn(clean.shape,device=device,dtype=clean.dtype,generator=generator))
    initial=copy.deepcopy(model.model).eval()
    with torch.no_grad():
        initial.core.output.weight.zero_();initial.core.output.bias.zero_()
    risks=[];baseline_risks=[]
    with torch.no_grad():
        for scale in scales:
            observed=clean+float(scale)*noise
            trained=CanonicalPosterior(model,scale)(observed)
            baseline=initial.canonical_posterior(observed,float(scale))
            risks.append(((trained-clean)/float(scale)).square().sum(1).cpu().numpy().reshape(8,128))
            baseline_risks.append(((baseline-clean)/float(scale)).square().sum(1).cpu().numpy().reshape(8,128))
    risks=np.asarray(risks);baseline_risks=np.asarray(baseline_risks)
    np.savez_compressed(dest/'denoising.npz',scales=scales,test_query_ids=np.arange(128),
        trained_risk=risks,initial_risk=baseline_risks)
    result['denoising']=[dict(lambda_value=float(scale),n=128,corruptions_per_query=8,
        trained_mean_risk=float(risks[j].mean()),initial_mean_risk=float(baseline_risks[j].mean()),
        paired_risk_difference_ci95=interval((risks[j]-baseline_risks[j]).mean(0))) for j,scale in enumerate(scales)]
    result['denoising_metric']='E sum-coordinate squared posterior error / RMS lambda squared, fresh native noise; lower is better; own-kernel risk, not identical observations across kernels'
    del initial
    # Trained, float64 posterior trace versus central differences in a complete
    # orthonormal output span; no Gaussian formula is used for either model.
    model.model.double();x=normalized[:3].double();basis=torch.linalg.qr(model.model.basis)[0]
    checks=[]
    for scale in sorted(set([float(scales[0]),selection['selected_lambda'],1.])):
        _,response=covariance_span_value_and_trace(model.model,x,scale)
        field=CanonicalPosterior(model,scale)
        estimates=[]
        for step in (1e-5,3e-6):
            finite=torch.zeros(len(x),device=device,dtype=torch.float64)
            with torch.no_grad():
                for j in range(basis.shape[1]):
                    direction=basis[:,j].expand_as(x)
                    finite+=((field(x+step*direction)-field(x-step*direction))*direction).sum(1)/(2*step)
            estimates.append(finite)
        gap=float((estimates[-1]-response).abs().max());refinement=float((estimates[0]-estimates[1]).abs().max())
        assert gap<2e-3 and refinement<2e-3,(done['variant'],cell.key,scale,gap,refinement)
        checks.append(dict(lambda_value=scale,finite_difference_gap=gap,step_refinement_gap=refinement))
    result.update(status='complete',seconds=time.time()-start,
        checkpoint_sha256=model.checkpoint_sha256,trained_derivative_checks=checks,
        response_selection_sha256=frozen,
        diagnostic_source_sha256=hashlib.sha256((dest/'evaluate_quality.py').read_bytes()).hexdigest())
    write_json(dest/'summary.json',result)
    print(json.dumps(dict(event='quality_complete',variant=done['variant'],cell=cell.key,
        response_lambda=selection['selected_lambda'],mae=result['response_metrics']['mae'],
        primary_lambda=done['selected_lambda'],primary_metrics=done['metrics'])),flush=True)
    del model,part,raw,loaded,test
    if str(device).startswith('cuda'):torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.run:return evaluate(args.run,args.device)
    if not args.watch:raise ValueError('provide --run or --watch')
    campaign=json.loads((BASE/'full_budget/campaign.json').read_text())
    pending={row['name'] for row in campaign['cells']}
    while pending:
        progress=False
        for name in sorted(pending):
            path=BASE/'full_budget'/name
            if (path/'complete.json').exists() and (BASE/'population/complete.json').exists():
                evaluate(path,args.device);pending.remove(name);progress=True
        if pending and not progress:
            if (BASE/'full_budget/launcher_complete.json').exists():
                status=json.loads((BASE/'full_budget/launcher_complete.json').read_text())
                if status['failures']:raise ValueError('campaign contains failed cells')
            time.sleep(15)
    write_json(BASE/'quality_complete.json',dict(status='complete',cells=len(campaign['cells'])))


if __name__=='__main__':main()
