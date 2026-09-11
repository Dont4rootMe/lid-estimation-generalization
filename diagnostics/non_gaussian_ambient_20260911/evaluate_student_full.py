"""Frozen-weight supplementary Student density readout, with holdout selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from datasets.registry import load_split
from experiments.fair_campaign import (verify_measurements, inventory, dataset_spec,
    write_json, file_sha)
from experiments import global_campaign as v1
from experiments import fair_measurements as measurement
from models import training
from student_full import density_dilation

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_gaussian_ambient_20260911'
SOURCES={name:Path(__file__).with_name(name).read_bytes() for name in
    ('student_full.py','evaluate_student_full.py','student_full_protocol.md')}
SOURCE_HASHES={name:hashlib.sha256(source).hexdigest() for name,source in SOURCES.items()}


def evaluate(run,device):
    start=time.time();done=verify_measurements(run/'complete.json');dest=run/'student_full'
    if (dest/'summary.json').exists():
        previous=json.loads((dest/'summary.json').read_text())
        assert previous['checkpoint_sha256']==done['checkpoint_sha256']
        assert previous['source_sha256']==SOURCE_HASHES
        return
    if dest.exists():dest.rename(run/f'student_full_partial_{time.time_ns()}')
    dest.mkdir()
    for name,source in SOURCES.items():(dest/name).write_bytes(source)
    cfg,cells=inventory();cell=next(c for c in cells if c.key==done['cell_key'])
    model=training.load_checkpoint(run/'model.pt',device=device)
    assert model.family in {'student_t_flow','pfgmpp'} and done['primary_readout']=='response'
    assert model.config.field_projection_rank is None and not hasattr(model.model,'basis')
    spec=dataset_spec(cell);data=BASE/'data/benchmarks'
    loaded=load_split(data,spec,'train',representation=cell.representation,mmap_mode='r')
    raw=loaded.features.reshape(len(loaded.features),-1)
    part=v1.partition_source_train(raw,loaded.lid,selection=cfg['campaign']['selection'],seed=0)
    assert v1._array_sha(part.selection_indices)==done['holdout_indices_sha256']
    scales=measurement.common_scales()
    def curve(features,response):
        query=(torch.tensor(np.asarray(features).copy(),dtype=torch.float32)
               -model.normalization_mean)/model.normalization_scale
        output=np.empty_like(response);extra=np.empty_like(response)
        for j,scale in enumerate(scales):
            for first in range(0,len(query),128):
                x=query[first:first+128].to(device)
                r=torch.as_tensor(response[first:first+128,j],device=device,dtype=x.dtype)
                f,b,contraction=density_dilation(model.model,x,float(scale),r,model.config.kernel_df)
                output[first:first+len(x),j]=f.cpu().numpy()
                extra[first:first+len(x),j]=contraction.cpu().numpy()
        assert np.isfinite(output).all() and np.isfinite(extra).all()
        return output,extra
    with np.load(run/'holdout_curve.npz') as z:
        holdout_r=z['prediction'];holdout_target=z['target']
    holdout,holdout_contraction=curve(part.selection_features,holdout_r)
    selection=measurement.supervised_common_selection(holdout,scales,holdout_target)
    receipt=dict(status='frozen',checkpoint_sha256=done['checkpoint_sha256'],
        selection=selection,readout='student_full',source_sha256=SOURCE_HASHES,
        source_train_holdout_only=True,primary_response_protocol_unchanged=True,
        supplementary_test_inference_started=False,
        timing='supplementary formula derived after training-prefix diagnostics; see protocol')
    write_json(dest/'selection.json',receipt);frozen=file_sha(dest/'selection.json')
    test=load_split(data,spec,'test',representation=cell.representation,mmap_mode='r')
    features=test.features.reshape(len(test.features),-1)
    with np.load(run/'test_scale_curves.npz') as z:test_r=z['common_prediction']
    prediction,test_contraction=curve(features,test_r)
    target=np.asarray(test.lid).reshape(-1);chosen=prediction[:,selection['selected_index']]
    automatic,indices,fallback=measurement.pointwise(prediction,scales,done['resolved']['geometry']['ambient_dim'])
    # The additional derivative is checked independently on three holdout
    # points. Full trace validity is already tested by the response campaign.
    query=(torch.tensor(np.asarray(part.selection_features[:3]).copy(),dtype=torch.float64)
        -model.normalization_mean.double())/model.normalization_scale
    query=query.to(device);model.model.double();checks=[]
    try:
        for scale in sorted(set([float(scales[0]),selection['selected_lambda'],1.])):
            f,b,contraction=density_dilation(model.model,query,scale,
                torch.zeros(len(query),device=device,dtype=query.dtype),model.config.kernel_df)
            h=1e-5
            with torch.no_grad():
                difference=(model.model(query,scale*np.exp(h))-model.model(query,scale*np.exp(-h)))/(2*h)
            finite=((query-b)*difference).sum(1)
            a=(model.config.kernel_df-2)*scale**2
            denominator=1+(query-b).square().sum(1)/a
            gap=float(((contraction-finite)/a/denominator).abs().max())
            assert gap<2e-3,(scale,gap)
            checks.append(dict(lambda_value=scale,n=3,extra_readout_finite_difference_gap=gap))
    finally:model.model.float()
    np.savez_compressed(dest/'curves.npz',scales=scales,holdout=holdout,
        holdout_response=holdout_r,holdout_target=holdout_target,
        holdout_query_ids=part.selection_indices,holdout_scale_contraction=holdout_contraction,
        test=prediction,test_response=test_r,test_target=target,test_query_ids=np.arange(len(target)),
        test_scale_contraction=test_contraction,kneedle_prediction=automatic,
        kneedle_index=indices,kneedle_fallback=fallback)
    assert file_sha(dest/'selection.json')==frozen
    result=dict(status='complete',readout='supplementary_student_full',variant=done['variant'],
        cell_key=cell.key,checkpoint_sha256=done['checkpoint_sha256'],source_sha256=SOURCE_HASHES,
        source_primary_complete_sha256=file_sha(run/'complete.json'),selection_sha256=frozen,
        curve_sha256=file_sha(dest/'curves.npz'),selection=selection,
        metrics=measurement.metric(chosen,target),automatic_metrics=measurement.metric(automatic,target),
        automatic_fallback_n=int(fallback.sum()),extra_derivative_checks=checks,seconds=time.time()-start)
    write_json(dest/'summary.json',result)
    print(json.dumps(dict(event='student_full_complete',variant=done['variant'],cell=cell.key,
        selected_lambda=selection['selected_lambda'],metrics=result['metrics'])),flush=True)
    del model,part,raw,loaded,test
    if str(device).startswith('cuda'):torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--watch',action='store_true')
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.run:return evaluate(args.run,args.device)
    if not args.watch:raise ValueError('provide --run or --watch')
    campaign=json.loads((BASE/'full_budget/campaign.json').read_text())
    pending={row['name'] for row in campaign['cells']}
    while pending:
        progress=False
        for name in sorted(pending):
            run=BASE/'full_budget'/name
            if (run/'complete.json').exists():
                evaluate(run,args.device);pending.remove(name);progress=True
        if pending and not progress:time.sleep(15)
    write_json(BASE/'student_full_complete.json',dict(status='complete',cells=len(campaign['cells'])))


if __name__=='__main__':main()
