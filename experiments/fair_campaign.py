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
from types import SimpleNamespace
import numpy as np
import torch

from datasets.registry import load_registry,apply_registry_overlay,load_split
from experiments import global_campaign as v1,global_campaign_v2 as v2
from experiments import fair_measurements as measurement
from experiments import fair_outputs as outputs
from experiments.fair_protocol import (ROOT,protocol,native_contracts,geometry,resolve,
    parameter_count,source_identity,digest,audit_group,capacities)
from models import training


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
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
    _,cells=inventory();rows=[];rules=protocol()
    for cell in cells:
        spec=dataset_spec(cell)
        geo=geometry(cell.dataset,cell.representation,spec.expected_shapes[cell.representation])
        cap=capacities(geo['kind'],geo['image_shape'][-1] if geo['kind']=='image' else geo['ambient_dim'])
        for variant in native_contracts():
            nf=variant=='scale_conditioned_nf'
            rows.append(dict(cell_key=cell.key,cell=asdict(cell),variant=variant,
                geometry=geo,architecture=('spatial RealNVP with U-Net conditioners' if geo['kind']=='image' else 'full-ambient RealNVP')
                    if variant=='scale_conditioned_nf' else ('shared U-Net' if geo['kind']=='image' else 'shared spectral residual'),
                backbone_width=cap['nf_width'] if nf else rules[geo['kind']+'_data']['width'],
                parameters=cap['nf_parameters'] if nf else cap['reference_parameters'],
                capacity_resolution=cap['nf_capacity_policy'] if nf else 'identical shared field core for every native family',
                protocol_sha256=digest(rules)))
    assert len(rows)==len(native_contracts())*len(v2.APPROVED_GLOBAL_CELL_KEYS)
    return dict(status='routing_audited',protocol=protocol(),rows=rows,
        cells=len(cells),native_interfaces=len(native_contracts()),trainings=len(rows),
        scope='configuration coverage; not completed benchmark training')


def prediction_spec(variant,geo):
    spec=dict(native_contracts()[variant])
    if variant=='scale_conditioned_nf':return spec
    spec['derivative_backend']='exact' if geo['kind']=='vector' else 'hutchinson'
    spec['trace_probes']=0 if geo['kind']=='vector' else protocol()['selection']['image_trace_probes']
    return spec


def select_scale(curve,scales,target,cell,reference=None):
    curve,scales=measurement.validate_curve(curve,scales,target)
    if cell.target_policy=='known_lid':
        receipt=measurement.supervised_common_selection(curve,scales,target)
        return receipt['selected_lambda'],receipt
    if cell.reference_dataset not in (None,cell.dataset):
        if reference is None:raise ValueError('dependent cell requires its completed same-method reference')
        status=measurement.validate_selection(reference['selected_lambda'],reference['selection'],reference=True)
        receipt=dict(status=status,
            criterion='reuse_reference_mean_kneedle',reference_dataset=cell.reference_dataset)
        if status=='selection_failed':
            receipt.update(failure_reason='reference_selection_failed',
                reference_failure_reason=reference['selection']['failure_reason'])
        return reference['selected_lambda'],receipt
    measurement.require_grid(scales,v2.unknown_reference_lambdas())
    index,receipt=v2.select_unknown_reference_kneedle(scales,curve)
    return (None if index is None else float(scales[index])),receipt


def validate_reference(reference,manifest,cell):
    expected_key=f'{cell.suite_id}/{cell.reference_dataset}/{cell.representation}'
    for key in ('variant','protocol_sha256','source_sha256','kind'):
        if reference[key]!=manifest[key]:raise ValueError('reference has mismatched '+key)
    if reference['cell_key']!=expected_key:raise ValueError('wrong reference dataset/representation')
    if reference['status']!='complete':raise ValueError('reference is not complete')
    measurement.validate_selection(reference['selected_lambda'],reference['selection'],reference=True)
    if 'config' in manifest and reference['steps_completed']!=manifest['config']['steps']:
        raise ValueError('reference has mismatched training budget')


def verify_measurements(path,row=None):
    """Replay receipts from saved arrays, including references and failed coverage."""
    measurement.validate_runtime()
    row=json.loads(path.read_text()) if row is None else row
    root=path.parent
    if row['status']!='complete':raise ValueError('measurement is not complete')
    for name,key in [('model.pt','checkpoint_sha256'),('selection.json','selection_receipt_sha256'),
                     ('holdout_curve.npz','holdout_curve_sha256'),('test_predictions.npz','test_predictions_sha256')]:
        if file_sha(root/name)!=row[key]:raise ValueError('changed measurement artifact: '+name)
    receipt=json.loads((root/'selection.json').read_text())
    for key,value in receipt.items():
        if key not in ('status','test_loaded') and row.get(key)!=value:
            raise ValueError('completion differs from frozen selection: '+key)
    if receipt['status']!='selected' or receipt['test_loaded'] is not False:
        raise ValueError('scale receipt was not frozen before test access')
    cell=SimpleNamespace(**receipt['cell']);known=cell.target_policy=='known_lid'
    if receipt['n_source_train']!=receipt['partition']['n_source_train'] or receipt['n_source_train']<receipt['fit_n']+receipt['holdout_n']:
        raise ValueError('source-train sample count differs from the partition')
    reference=None
    if not known and cell.reference_dataset not in (None,cell.dataset):
        ref_path=root/'reference_complete.json'
        if file_sha(ref_path)!=receipt['reference_receipt_sha256']:raise ValueError('reference receipt changed')
        reference=json.loads(ref_path.read_text());validate_reference(reference,receipt,cell)
        if file_sha(root/'reference_test_predictions.npz')!=reference['test_predictions_sha256']:
            raise ValueError('reference test predictions changed')
    with np.load(root/'holdout_curve.npz',allow_pickle=False) as z:
        curve,scales=z['prediction'],z['scales'];target=z['target'] if known else None
        if len(curve)!=receipt['holdout_n'] or v1._array_sha(z['query_ids'])!=receipt['effective_holdout_indices_sha256']:
            raise ValueError('holdout query identity differs')
        selected,selection=select_scale(curve,scales,target,cell,reference)
        if selected!=receipt['selected_lambda'] or selection!=receipt['selection']:
            raise ValueError('selection does not replay from the saved holdout')
        plan=measurement.known_plan(curve,scales,target) if known else None
        if plan!=receipt['measurement_plan']:raise ValueError('measurement plan differs from the saved holdout')
    status=measurement.validate_selection(selected,selection,reference=not known)
    if row['measurement_status']!=status:raise ValueError('measurement coverage differs')
    primary=receipt['primary_readout']
    if primary!=outputs.readouts(receipt['variant'])[0]:raise ValueError('incorrect primary readout')
    readouts=list(outputs.readouts(receipt['variant'])) if selected is not None else []
    with np.load(root/'test_predictions.npz',allow_pickle=False) as z:
        if set(z.files)!=set(readouts+['query_ids','target','labels']):raise ValueError('missing or extra test readouts')
        if not np.array_equal(z['query_ids'],np.arange(row['test_n'],dtype=np.int64)):
            raise ValueError('test query identity differs')
        if v1._array_sha(z['labels'])!=row['test_labels_sha256']:
            raise ValueError('test class labels differ')
        if z['labels'].size and (z['labels'].shape!=(row['test_n'],) or z['labels'].dtype.kind not in 'iu'):
            raise ValueError('invalid test class labels')
        if cell.target_policy=='paired_delta' and z['labels'].shape!=(row['test_n'],):
            raise ValueError('paired-delta class labels are missing')
        target=z['target'] if known else None
        if known and target.shape!=(row['test_n'],):raise ValueError('test targets differ')
        expected={}
        for readout in readouts:
            if len(z[readout])!=row['test_n']:raise ValueError('test prediction count differs')
            expected[readout]=measurement.metric(z[readout],target,
                'float64' if readout=='fixed_likelihood' else 'float32')
        if row['metrics']!=expected:raise ValueError('reported metrics differ from saved predictions')
        primary_values=z[primary] if selected is not None else None
        if not known:
            if reference is None:
                relative=outputs.reference_metrics(row,z,row,z)
            else:
                with np.load(root/'reference_test_predictions.npz',allow_pickle=False) as base:
                    relative=outputs.reference_metrics(row,z,reference,base)
            if row['reference_metrics']!=relative:raise ValueError('reference metrics differ from saved predictions')
        elif row['reference_metrics']:
            raise ValueError('unexpected relative metrics on known-LID data')
    if known:
        curve_path=root/'test_scale_curves.npz'
        if file_sha(curve_path)!=row['test_scale_curves_sha256']:raise ValueError('test scale curves changed')
        with np.load(curve_path,allow_pickle=False) as z:
            expected,arrays=measurement.known_results(z['common_prediction'],z['legacy_prediction'],
                target,plan,receipt['resolved']['geometry']['ambient_dim'])
            if set(arrays)!=set(z.files) or any(not np.array_equal(z[k],v) for k,v in arrays.items()):
                raise ValueError('pointwise decisions do not replay from saved curves')
            index=int(np.argmin(abs(z['common_scales']-selected)))
            if not np.array_equal(primary_values,z['common_prediction'][:,index]):
                raise ValueError('primary prediction differs from its saved scale curve')
        if row['diagnostic_metrics']!=expected:raise ValueError('automatic metrics differ from saved curves')
        if row['automatic_metrics']!=expected[plan['primary_automatic_metric']]:
            raise ValueError('primary automatic metrics must report common-grid Kneedle')
    elif row['diagnostic_metrics'] or row['test_scale_curves_sha256'] is not None:
        raise ValueError('unexpected known-LID diagnostics on an unknown-LID cell')
    elif row['automatic_metrics']:
        raise ValueError('unexpected pointwise automatic metrics on an unknown-LID cell')
    return row


def run(args):
    measurement.validate_runtime()
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
    rank=geo['ambient_dim'] if geo['kind']=='vector' else None
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
        n_source_train=len(raw),partition=part.record,
        fit_n=len(fit),holdout_n=len(holdout),created_unix=time.time())
    reference=None
    if cell.target_policy!='known_lid' and cell.reference_dataset not in (None,cell.dataset):
        if args.reference is None:raise ValueError('provide --reference before training a dependent cell')
        reference=verify_measurements(args.reference);validate_reference(reference,manifest,cell)
        manifest['reference_receipt_sha256']=file_sha(args.reference)
    out.mkdir(parents=True)
    if reference is not None:
        (out/'reference_complete.json').write_bytes(args.reference.read_bytes())
        if file_sha(out/'reference_complete.json')!=manifest['reference_receipt_sha256']:
            raise ValueError('reference changed during preparation')
        (out/'reference_test_predictions.npz').write_bytes((args.reference.parent/'test_predictions.npz').read_bytes())
        if file_sha(out/'reference_test_predictions.npz')!=reference['test_predictions_sha256']:
            raise ValueError('reference predictions changed during preparation')
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
    primary=outputs.readouts(args.variant)[0]
    known=cell.target_policy=='known_lid'
    scales=measurement.common_scales() if known else v2.unknown_reference_lambdas()
    def predict(query,grid,readout=primary):
        return v2._prediction_curve(training.predict_lid,result,query,np.asarray(grid),model=model_spec,
            seed=0,batch_size=128,readout=readout)
    curve=predict(holdout,scales)
    selected,selection=select_scale(curve,scales,target,cell,reference)
    status=measurement.validate_selection(selected,selection,reference=not known)
    plan=measurement.known_plan(curve,scales,target) if known else None
    np.savez_compressed(out/'holdout_curve.npz',scales=scales,prediction=curve,
        target=np.asarray([] if target is None else target),query_ids=part.selection_indices[:len(holdout)])
    receipt=dict(manifest,status='selected',selection=selection,selected_lambda=selected,
        checkpoint_sha256=result.checkpoint_sha256,normalization_sha256=result.preprocessing_sha256,
        actual_config=result.config.to_dict(),best_step=result.best_epoch,
        steps_completed=result.metrics['steps_completed'],native_validation_loss=result.best_validation_loss,
        train_seconds=time.monotonic()-start,primary_readout=primary,measurement_plan=plan,
        holdout_curve_sha256=file_sha(out/'holdout_curve.npz'),test_loaded=False)
    write_json(out/'selection.json',receipt)
    selection_hash=file_sha(out/'selection.json')
    test=load_split(data_root,spec,'test',representation=cell.representation,mmap_mode='r')
    queries=test.features.reshape(len(test.features),-1)
    if args.preflight:queries=queries[:min(16,len(queries))]
    targets=np.asarray(test.lid).reshape(-1)[:len(queries)] if known and test.lid is not None else None
    labels=np.asarray([],dtype=np.int64) if test.labels is None else np.asarray(test.labels).reshape(-1)[:len(queries)]
    metrics={};predictions=dict(query_ids=np.arange(len(queries),dtype=np.int64),labels=labels,
        target=np.asarray([] if targets is None else targets));diagnostics={}
    if known:
        common_curve=predict(queries,scales)
        legacy_curve=predict(queries,measurement.legacy_scales())
        diagnostics,arrays=measurement.known_results(common_curve,legacy_curve,targets,plan,geo['ambient_dim'])
        np.savez_compressed(out/'test_scale_curves.npz',**arrays)
    if selected is not None:
        readouts=outputs.readouts(args.variant)
        for readout in readouts:
            if args.variant=='scale_conditioned_nf' and readout=='fixed_likelihood':result.model.double()
            values=common_curve[:,int(np.argmin(abs(scales-selected)))] if known and readout==primary else predict(queries,[selected],readout)[:,0]
            predictions[readout]=values
            metrics[readout]=measurement.metric(values,targets,
                'float64' if args.variant=='scale_conditioned_nf' and readout=='fixed_likelihood' else 'float32')
    np.savez_compressed(out/'test_predictions.npz',**predictions)
    if file_sha(out/'selection.json')!=selection_hash:raise ValueError('selection changed during test inference')
    final=dict(receipt,status='complete',measurement_status=status,selection_receipt_sha256=selection_hash,
        test_n=len(queries),test_labels_sha256=v1._array_sha(labels),metrics=metrics,diagnostic_metrics=diagnostics,
        automatic_metrics=diagnostics[plan['primary_automatic_metric']] if known else {},
        test_files_sha256={p.name:file_sha(p) for p in test.source_paths.values()},
        test_predictions_sha256=file_sha(out/'test_predictions.npz'),
        test_scale_curves_sha256=file_sha(out/'test_scale_curves.npz') if known else None,
        total_seconds=time.monotonic()-start,test_loaded=True,reload_exact=True)
    final['reference_metrics']={}
    if not known:
        if reference is None:
            final['reference_metrics']=outputs.reference_metrics(final,predictions,final,predictions)
        else:
            with np.load(out/'reference_test_predictions.npz',allow_pickle=False) as base:
                final['reference_metrics']=outputs.reference_metrics(final,predictions,reference,base)
    write_json(out/'complete.pending.json',final)
    verify_measurements(out/'complete.pending.json')
    (out/'complete.pending.json').replace(out/'complete.json')
    print(json.dumps(dict(status='complete',measurement_status=status,kind=final['kind'],
        variant=args.variant,cell=cell.key,metrics=metrics,automatic_metrics=final['automatic_metrics'])),flush=True)


def aggregate(root,scope='all'):
    plan=matrix();expected={(r['cell_key'],r['variant']) for r in plan['rows']
        if scope=='all' or r['cell']['target_policy']=='known_lid'}
    declared={(r['cell_key'],r['variant']):r for r in plan['rows']}
    rows=[];entries=[];seen=set();receipt_hashes={}
    for path in root.rglob('complete.json'):
        row=json.loads(path.read_text())
        if 'protocol_sha256' not in row:raise ValueError('unrecognized historical result in aggregate root')
        key=(row['cell_key'],row['variant'])
        if key not in expected:continue
        if key in seen:raise ValueError('duplicate trained result for a comparison cell')
        if row['kind']!='benchmark' or row['steps_completed']!=protocol()['training']['steps']:
            raise ValueError('preflight or incomplete budget cannot enter the benchmark')
        if row['cell']!=declared[key]['cell'] or row['resolved']['geometry']!=declared[key]['geometry']:
            raise ValueError('cell identity or geometry differs from the declared inventory')
        if row['n_source_train']!=dataset_spec(SimpleNamespace(**row['cell'])).expected_samples['train']:
            raise ValueError('source-train count differs from the declared inventory')
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
        verify_measurements(path,row)
        receipt_hashes[key]=file_sha(path)
        seen.add(key);rows.append(row);entries.append((row,path.parent))
    if seen!=expected:raise ValueError(f'incomplete fair matrix: {len(seen)}/{len(expected)} trained cells')
    for row in rows:
        if 'reference_receipt_sha256' in row:
            cell=row['cell'];key=(f"{cell['suite_id']}/{cell['reference_dataset']}/{cell['representation']}",row['variant'])
            if receipt_hashes.get(key)!=row['reference_receipt_sha256']:
                raise ValueError('dependent result does not use the aggregated reference receipt')
    for key in {key for key,_ in expected}:
        group=[r for r in rows if r['cell_key']==key];audit_group(group)
        for field in ('train_files_sha256','test_files_sha256','fit_indices_sha256','holdout_indices_sha256',
                      'normalization_sha256','fit_n','holdout_n','test_n','n_source_train','test_labels_sha256'):
            if any(r[field]!=group[0][field] for r in group):raise ValueError('data/query mismatch: '+field)
    coverage={s:sum(r['measurement_status']==s for r in rows)
        for s in ('selected','scale_unresolved','selection_failed')}
    records=outputs.result_records(entries)
    cells={r['cell_key']:r['cell'] for r in plan['rows'] if (r['cell_key'],r['variant']) in expected}
    scores=outputs.table_scores(records,list(cells.values()))
    return dict(status='complete' if coverage['selected']==len(rows) else 'complete_with_selection_issues',
        coverage=coverage,scope=scope,protocol_sha256=digest(protocol()),trainings=len(rows),rows=rows,
        result_records=records,table_scores=scores,
        score_coverage={s:sum(r['status']==s for r in scores) for s in sorted({r['status'] for r in scores})})


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
    v=sub.add_parser('verify');v.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    if args.command=='matrix':
        value=matrix();write_json(args.output,value);print(json.dumps({k:v for k,v in value.items() if k not in ('rows','protocol')}))
    elif args.command=='aggregate':
        value=aggregate(args.root,args.scope)
        write_json(args.output,value)
        outputs.write_csv(args.output.with_name(args.output.stem+'.results.csv'),value['result_records'])
        outputs.write_csv(args.output.with_name(args.output.stem+'.scores.csv'),value['table_scores'])
    elif args.command=='verify':
        row=verify_measurements(args.receipt)
        print(json.dumps(dict(status='verified',kind=row['kind'],measurement_status=row['measurement_status'])))
    else:run(args)


if __name__=='__main__':main()
