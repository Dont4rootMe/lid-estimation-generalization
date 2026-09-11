"""Dataset-routed native comparison: matrix audit, one-cell execution, aggregation.

No historical checkpoint is accepted as a new-protocol result. Data preparation
and training are separated from test access; all methods use the same heldout
indices, optimizer policy, physical scale selector and query sets on a cell.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from datasets.registry import load_registry,apply_registry_overlay,load_split
from experiments import global_campaign as v1,global_campaign_v2 as v2
from experiments.fair_protocol import (ROOT,protocol,native_contracts,geometry,resolve,
    parameter_count,source_identity,digest,audit_group)
from models import training


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def inventory():
    cfg=v2.compose_global_campaign_v2_config(root=ROOT)
    return cfg,v1.load_campaign_inventory(cfg,ROOT)


def dataset_spec(cell):
    registry=load_registry(ROOT/cell.registry,validate_official_coverage=False)
    if cell.registry_overlay:
        registry=apply_registry_overlay(registry,ROOT/cell.registry_overlay)
    return registry[cell.dataset]


def matrix():
    _,cells=inventory();rows=[]
    for cell in cells:
        spec=dataset_spec(cell)
        geo=geometry(cell.dataset,cell.representation,spec.expected_shapes[cell.representation])
        for variant in native_contracts():
            rows.append(dict(cell_key=cell.key,cell=asdict(cell),variant=variant,
                geometry=geo,architecture=('spatial RealNVP with U-Net conditioners' if geo['kind']=='image' else 'covariance RealNVP')
                    if variant=='scale_conditioned_nf' else ('shared U-Net' if geo['kind']=='image' else 'shared spectral residual'),
                capacity_resolution='exact image parameter count' if geo['kind']=='image' else 'resolve from optimizer-fit covariance before training',
                protocol_sha256=digest(protocol())))
    assert len(rows)==len(v2.APPROVED_MODEL_VARIANTS)*len(v2.APPROVED_GLOBAL_CELL_KEYS)
    return dict(status='routing_audited',protocol=protocol(),rows=rows,
        cells=len(cells),native_interfaces=len(native_contracts()),trainings=len(rows),
        scope='configuration coverage; not completed benchmark training')


def prediction_spec(variant,geo):
    spec=dict(native_contracts()[variant])
    if variant=='scale_conditioned_nf':return spec
    spec['derivative_backend']='active_exact' if geo['kind']=='vector' else 'hutchinson'
    spec['trace_probes']=0 if geo['kind']=='vector' else protocol()['selection']['image_trace_probes']
    return spec


def select_scale(curve,scales,target,cell,reference=None):
    if cell.target_policy=='known_lid':
        initial=v2.initial_supervised_lambdas()
        def values(requested):
            ids=[int(np.argmin(abs(scales-s))) for s in requested]
            if not np.allclose(scales[ids],requested,rtol=1e-12,atol=1e-12):raise ValueError('selector requested an unavailable scale')
            return curve[:,ids]
        grid,_,index,receipt=v2.select_supervised_bounded(initial,values(initial),target,evaluate=values)
        return float(grid[index]),receipt
    if cell.reference_dataset not in (None,cell.dataset):
        if reference is None:raise ValueError('dependent cell requires its completed same-method reference')
        return reference['selected_lambda'],dict(status=reference['selection']['status'],
            criterion='reuse_reference_mean_kneedle',reference_dataset=cell.reference_dataset)
    index,receipt=v2.select_unknown_reference_kneedle(scales,curve)
    return (None if index is None else float(scales[index])),receipt


def validate_reference(reference,manifest,cell):
    expected_key=f'{cell.suite_id}/{cell.reference_dataset}/{cell.representation}'
    for key in ('variant','protocol_sha256','source_sha256','kind'):
        if reference[key]!=manifest[key]:raise ValueError('reference has mismatched '+key)
    if reference['cell_key']!=expected_key:raise ValueError('wrong reference dataset/representation')
    if reference['status']!='complete':raise ValueError('reference is not complete')


def run(args):
    cfg,cells=inventory()
    cell=next((c for c in cells if c.key==args.cell),None)
    if cell is None:raise ValueError('cell is not in the fixed full campaign inventory')
    if args.variant not in native_contracts():raise ValueError('unknown native interface')
    out=args.output
    if out.exists():raise FileExistsError('choose a fresh cell output directory')
    spec=dataset_spec(cell)
    data_root=args.canonical_root if cell.source_kind=='exact_archive' else args.generated_root
    if data_root is None:raise ValueError('provide the data root for this source kind')
    # Test split is not loaded until selection.json has been written below.
    loaded=load_split(data_root,spec,'train',representation=cell.representation,mmap_mode='r')
    raw=loaded.features.reshape(len(loaded.features),-1)
    geo=geometry(cell.dataset,cell.representation,loaded.feature_shape)
    part=v1.partition_source_train(raw,loaded.lid,selection=cfg['campaign']['selection'],seed=0)
    fit,holdout=part.fit_features,part.selection_features
    target=part.selection_target
    if args.preflight:
        fit=fit[:min(len(fit),1024)];holdout=holdout[:min(len(holdout),16)]
        if target is not None:target=target[:len(holdout)]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rank=None
    if geo['kind']=='vector':
        from models.preconditioned_field import training_covariance_span
        tensor=training._flat_finite_data(fit,name='fit')
        mean,rms,_,_=training._normalization(tensor,enabled=True,epsilon=1e-8)
        rank=training_covariance_span((tensor-mean)/rms,args.device)[0].shape[1]
    config,resolved=resolve(args.variant,geo,rank=rank,device=args.device,
        steps=args.steps,preflight=args.preflight)
    sources=source_identity()
    manifest=dict(variant=args.variant,cell_key=cell.key,cell=asdict(cell),resolved=resolved,
        config=config.to_dict(),kind=resolved['kind'],protocol_sha256=digest(protocol()),
        source_sha256=sources,train_files_sha256={p.name:file_sha(p) for p in loaded.source_paths.values()},
        registry_sha256=file_sha(ROOT/cell.registry),registry_overlay_sha256=file_sha(ROOT/cell.registry_overlay) if cell.registry_overlay else None,
        fit_indices_sha256=v1._array_sha(part.fit_indices),holdout_indices_sha256=v1._array_sha(part.selection_indices),
        effective_fit_indices_sha256=v1._array_sha(part.fit_indices[:len(fit)]),
        effective_holdout_indices_sha256=v1._array_sha(part.selection_indices[:len(holdout)]),
        fit_n=len(fit),holdout_n=len(holdout),created_unix=time.time())
    reference=None
    if cell.target_policy!='known_lid' and cell.reference_dataset not in (None,cell.dataset):
        if args.reference is None:raise ValueError('provide --reference before training a dependent cell')
        reference=json.loads(args.reference.read_text());validate_reference(reference,manifest,cell)
        manifest['reference_receipt_sha256']=file_sha(args.reference)
    out.mkdir(parents=True)
    write_json(out/'manifest.json',manifest)
    for name in sources:
        destination=out/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    start=time.monotonic()
    def callback(row):
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(dict(row,elapsed_seconds=time.monotonic()-start))+'\n')
    family=native_contracts()[args.variant]['family']
    result=training.train_model(family,fit,holdout,config,out/'model.pt',log_callback=callback,
        progress_checkpoint_path=out/'progress.pt')
    if parameter_count(result.model)!=resolved['actual_parameters']:raise ValueError('trained capacity differs from the common contract')
    reloaded=training.load_checkpoint(out/'model.pt',device=args.device)
    for key,value in result.model.state_dict().items():torch.testing.assert_close(value,reloaded.model.state_dict()[key],rtol=0,atol=0)
    if sources!=source_identity():raise ValueError('source changed during this cell')
    model_spec=prediction_spec(args.variant,geo)
    primary='ols5' if args.variant=='scale_conditioned_nf' else 'full'
    scales=2.**(np.arange(-16,13)/2) if cell.target_policy=='known_lid' else v2.unknown_reference_lambdas()
    def predict(query,grid,readout=primary):
        return v2._prediction_curve(training.predict_lid,result,query,np.asarray(grid),model=model_spec,
            seed=0,batch_size=128,readout=readout)
    curve=predict(holdout,scales)
    selected,selection=select_scale(curve,scales,target,cell,reference)
    np.savez_compressed(out/'holdout_curve.npz',scales=scales,prediction=curve,
        target=np.asarray([] if target is None else target))
    receipt=dict(manifest,status='selected',selection=selection,selected_lambda=selected,
        checkpoint_sha256=result.checkpoint_sha256,normalization_sha256=result.preprocessing_sha256,
        actual_config=result.config.to_dict(),best_step=result.best_epoch,
        steps_completed=result.metrics['steps_completed'],native_validation_loss=result.best_validation_loss,
        train_seconds=time.monotonic()-start,primary_readout=primary,
        holdout_curve_sha256=file_sha(out/'holdout_curve.npz'),test_loaded=False)
    write_json(out/'selection.json',receipt)
    selection_hash=file_sha(out/'selection.json')
    test=load_split(data_root,spec,'test',representation=cell.representation,mmap_mode='r')
    queries=test.features.reshape(len(test.features),-1)
    if args.preflight:queries=queries[:min(16,len(queries))]
    metrics={};predictions={}
    if selected is not None:
        readouts=[primary,'fixed_likelihood'] if args.variant=='scale_conditioned_nf' else ['full']
        for readout in readouts:
            if args.variant=='scale_conditioned_nf' and readout=='fixed_likelihood':result.model.double()
            values=predict(queries,[selected],readout)[:,0];predictions[readout]=values
            metrics[readout]=dict(mean=float(values.mean()),n=len(values),
                dtype='float64' if args.variant=='scale_conditioned_nf' and readout=='fixed_likelihood' else 'float32')
            if test.lid is not None:
                targets=np.asarray(test.lid).reshape(-1)[:len(values)]
                metrics[readout]['mae']=float(np.abs(values-targets).mean())
        np.savez_compressed(out/'test_predictions.npz',**predictions)
    if file_sha(out/'selection.json')!=selection_hash:raise ValueError('selection changed during test inference')
    final=dict(receipt,status='complete',selection_receipt_sha256=selection_hash,test_n=len(queries),metrics=metrics,
        test_files_sha256={p.name:file_sha(p) for p in test.source_paths.values()},
        test_predictions_sha256=file_sha(out/'test_predictions.npz') if selected is not None else None,
        total_seconds=time.monotonic()-start,test_loaded=True,reload_exact=True)
    write_json(out/'complete.json',final)
    print(json.dumps(dict(status='complete',kind=final['kind'],variant=args.variant,cell=cell.key,metrics=metrics)),flush=True)


def aggregate(root,scope='all'):
    plan=matrix();expected={(r['cell_key'],r['variant']) for r in plan['rows']
        if scope=='all' or r['cell']['target_policy']=='known_lid'}
    rows=[];seen=set()
    for path in root.rglob('complete.json'):
        row=json.loads(path.read_text())
        if 'protocol_sha256' not in row:raise ValueError('unrecognized historical result in aggregate root')
        key=(row['cell_key'],row['variant'])
        if key not in expected:continue
        if key in seen:raise ValueError('duplicate trained result for a comparison cell')
        if row['kind']!='benchmark' or row['steps_completed']!=protocol()['training']['steps']:
            raise ValueError('preflight or incomplete budget cannot enter the benchmark')
        if row['protocol_sha256']!=digest(protocol()) or row['source_sha256']!=source_identity():
            raise ValueError('result belongs to a different protocol/source revision')
        expected_config,expected_resolution=resolve(row['variant'],row['resolved']['geometry'],
            rank=row['resolved']['rank'],device=row['config']['device'])
        if digest(row['resolved'])!=digest(expected_resolution) or digest(row['actual_config'])!=digest(expected_config.to_dict()):
            raise ValueError('actual configuration differs from the frozen comparison policy')
        if row['status']!='complete' or file_sha(path.parent/'model.pt')!=row['checkpoint_sha256']:
            raise ValueError('missing or altered producing checkpoint')
        if file_sha(path.parent/'selection.json')!=row['selection_receipt_sha256']:
            raise ValueError('selection receipt changed')
        if file_sha(path.parent/'holdout_curve.npz')!=row['holdout_curve_sha256']:
            raise ValueError('holdout predictions changed')
        if row['test_predictions_sha256'] is not None:
            if file_sha(path.parent/'test_predictions.npz')!=row['test_predictions_sha256']:
                raise ValueError('test predictions changed')
        seen.add(key);rows.append(row)
    if seen!=expected:raise ValueError(f'incomplete fair matrix: {len(seen)}/{len(expected)} trained cells')
    for key in {key for key,_ in expected}:
        group=[r for r in rows if r['cell_key']==key];audit_group(group)
        for field in ('train_files_sha256','test_files_sha256','fit_indices_sha256','holdout_indices_sha256',
                      'normalization_sha256','fit_n','holdout_n','test_n'):
            if any(r[field]!=group[0][field] for r in group):raise ValueError('data/query mismatch: '+field)
    return dict(status='complete',scope=scope,protocol_sha256=digest(protocol()),trainings=len(rows),rows=rows)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    m=sub.add_parser('matrix');m.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--variant',required=True);r.add_argument('--cell',required=True)
    r.add_argument('--canonical-root',type=Path);r.add_argument('--generated-root',type=Path)
    r.add_argument('--output',type=Path,required=True);r.add_argument('--device',default='cuda')
    r.add_argument('--steps',type=int);r.add_argument('--preflight',action='store_true');r.add_argument('--reference',type=Path)
    a=sub.add_parser('aggregate');a.add_argument('--root',type=Path,required=True)
    a.add_argument('--scope',choices=['all','known_lid'],default='all');a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='matrix':
        value=matrix();write_json(args.output,value);print(json.dumps({k:v for k,v in value.items() if k not in ('rows','protocol')}))
    elif args.command=='aggregate':write_json(args.output,aggregate(args.root,args.scope))
    else:run(args)


if __name__=='__main__':main()
