"""Negative controls for missing automatic tracks and inconsistent receipts."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import fair_measurements as m,global_campaign_v2 as v2
from experiments.fair_campaign import (select_scale,validate_reference,verify_measurements,
    write_json,file_sha)
from experiments.global_campaign import _array_sha


def test_legacy_pointwise_matches_pinned_flipd():
    rng=np.random.default_rng(18)
    times=v2.flipd_timesteps()
    curves=np.concatenate([rng.normal(size=(48,22)),
        (1/(times+.02))[None,:],np.ones((1,22))])
    with np.errstate(invalid='ignore'):
        old,t,failed=v2.flipd_kneedle_pointwise(curves,30)
    values,index,fallback=m.pointwise(curves,m.legacy_scales(),30)
    np.testing.assert_array_equal(values,old)
    np.testing.assert_array_equal(fallback,failed)
    np.testing.assert_array_equal(times[index[~fallback]],t[~failed])
    assert values[-1]==30 and fallback[-1]


@pytest.mark.parametrize('case',['empty','nan','unordered','duplicate','negative','target'])
def test_invalid_measurement_inputs_are_rejected(case):
    scales=m.common_scales();curve=np.ones((4,len(scales)));target=np.ones(4)
    if case=='empty':curve=curve[:0];target=target[:0]
    elif case=='nan':curve[0,0]=np.nan
    elif case=='unordered':scales=scales[::-1]
    elif case=='duplicate':scales[0]=scales[1]
    elif case=='negative':scales[0]=-1
    else:target=target[:2]
    with pytest.raises(ValueError):m.validate_curve(curve,scales,target)


def test_failed_reference_retains_its_reason():
    cell=SimpleNamespace(target_policy='sample_size',dataset='step2',reference_dataset='step1')
    ref=dict(selected_lambda=None,selection=dict(status='selection_failed',failure_reason='no_knee'))
    scale,receipt=select_scale(np.ones((4,50)),v2.unknown_reference_lambdas(),None,cell,ref)
    assert scale is None and receipt['status']=='selection_failed'
    assert receipt['failure_reason']=='reference_selection_failed'
    assert receipt['reference_failure_reason']=='no_knee'


@pytest.mark.parametrize('scale,status,reason',[(999.,'selected',None),
    (None,'selected',None),(np.nan,'selected',None),(1.,'selection_failed','no_knee'),
    (None,'selection_failed',None),(1.,'unknown',None)])
def test_invalid_reference_status_and_scale_cannot_propagate(scale,status,reason):
    cell=SimpleNamespace(suite_id='e1',reference_dataset='step1',representation='dataset')
    manifest=dict(variant='ve_diffusion',protocol_sha256='a',source_sha256={'x':'b'},kind='benchmark')
    ref=dict(manifest,cell_key='e1/step1/dataset',status='complete',selected_lambda=scale,
        selection=dict(status=status,failure_reason=reason))
    with pytest.raises(ValueError):validate_reference(ref,manifest,cell)


def test_common_grid_supervised_comparator_is_frozen_before_test_labels():
    scales=m.common_scales();target=np.arange(4,dtype=float)
    holdout=target[:,None]+np.log2(scales)[None,:]**2
    plan=m.known_plan(holdout,scales,target)
    assert plan['supervised_common_grid']['selected_lambda']==1.
    assert plan['test_labels_used_for_selection'] is False
    test_curve=holdout[:,::-1]
    a,_=m.known_results(test_curve,np.ones((4,22)),target,plan,30)
    b,_=m.known_results(test_curve,np.ones((4,22)),target+100,plan,30)
    assert a['supervised_common_grid']['selected_lambda']==b['supervised_common_grid']['selected_lambda']==1.
    assert a['kneedle_common_grid']['selected_lambda_counts']==b['kneedle_common_grid']['selected_lambda_counts']
    assert a['kneedle_legacy']['fallback_n']==4
    assert a['kneedle_legacy']['mae']==np.abs(30-target).mean()
    assert a['supervised_common_grid']['mae']!=b['supervised_common_grid']['mae']


def make_receipt(root,known=True,reference=None):
    """Small synthetic artifacts exercise accounting, not model-quality claims."""
    root.mkdir()
    grid=m.common_scales() if known else v2.unknown_reference_lambdas()
    target=np.arange(4,dtype=float) if known else None
    curve=(target[:,None]+np.log2(grid)[None,:]**2) if known else np.ones((4,50))
    cell=SimpleNamespace(target_policy='known_lid' if known else 'sample_size',
        suite_id='e6' if known else 'e1',dataset='fixture' if known else ('step2' if reference else 'step1'),
        representation='coefficients',reference_dataset=None if known else 'step1',expected_lid_delta=0.)
    ref=None
    if reference:
        (root/'reference_complete.json').write_bytes(reference.read_bytes())
        ref=json.loads(reference.read_text())
        (root/'reference_test_predictions.npz').write_bytes((reference.parent/'test_predictions.npz').read_bytes())
    selected,selection=select_scale(curve,grid,target,cell,ref)
    plan=m.known_plan(curve,grid,target) if known else None
    np.savez_compressed(root/'holdout_curve.npz',prediction=curve,scales=grid,
        target=np.asarray([] if target is None else target),query_ids=np.arange(4,dtype=np.int64))
    (root/'model.pt').write_bytes(b'test fixture checkpoint')
    receipt=dict(status='selected',variant='ve_diffusion',cell_key=f'{cell.suite_id}/{cell.dataset}/coefficients',
        cell=vars(cell),kind='preflight',protocol_sha256='fixture',source_sha256={'fixture':'fixture'},
        resolved=dict(geometry=dict(ambient_dim=30)),config=dict(steps=2),steps_completed=2,
        n_source_train=20,partition=dict(n_source_train=20),fit_n=16,
        checkpoint_sha256=file_sha(root/'model.pt'),selected_lambda=selected,selection=selection,
        holdout_n=4,effective_holdout_indices_sha256=_array_sha(np.arange(4,dtype=np.int64)),
        holdout_curve_sha256=file_sha(root/'holdout_curve.npz'),primary_readout='full',
        measurement_plan=plan,test_loaded=False)
    if reference:receipt['reference_receipt_sha256']=file_sha(reference)
    write_json(root/'selection.json',receipt)
    pred=dict(query_ids=np.arange(4,dtype=np.int64),labels=np.asarray([],dtype=np.int64),
        target=np.asarray([] if target is None else target))
    metrics={};diagnostics={}
    if selected is not None:
        pred['full']=curve[:,int(np.argmin(abs(grid-selected)))]
        metrics['full']=m.metric(pred['full'],target)
    if known:
        diagnostics,arrays=m.known_results(curve,np.ones((4,22)),target,plan,30)
        np.savez_compressed(root/'test_scale_curves.npz',**arrays)
    np.savez_compressed(root/'test_predictions.npz',**pred)
    final=dict(receipt,status='complete',measurement_status=selection['status'],test_loaded=True,
        selection_receipt_sha256=file_sha(root/'selection.json'),test_n=4,
        test_labels_sha256=_array_sha(pred['labels']),metrics=metrics,reference_metrics={},
        test_files_sha256={'dataset.npy':'fixture'},
        automatic_metrics=diagnostics[m.PRIMARY_AUTOMATIC_METRIC] if known else {},
        diagnostic_metrics=diagnostics,test_predictions_sha256=file_sha(root/'test_predictions.npz'),
        test_scale_curves_sha256=file_sha(root/'test_scale_curves.npz') if known else None)
    path=root/'complete.json';write_json(path,final)
    verify_measurements(path)
    return path,final


@pytest.mark.parametrize('case',['metric','selected_scale','coverage','automatic_metric','automatic_missing','curve_changed'])
def test_complete_receipt_cannot_override_frozen_arrays(tmp_path,case):
    path,row=make_receipt(tmp_path/'known')
    if case=='metric':row['metrics']['full']['mae']=999.
    elif case=='selected_scale':row['selected_lambda']=64.
    elif case=='coverage':row['measurement_status']='selection_failed'
    elif case=='automatic_metric':row['diagnostic_metrics']['kneedle_legacy']['mae']=0.
    elif case=='automatic_missing':row['diagnostic_metrics']={}
    else:
        with np.load(path.parent/'test_scale_curves.npz') as z:arrays={k:z[k] for k in z.files}
        arrays['kneedle_legacy_prediction'][0]=0.
        np.savez_compressed(path.parent/'test_scale_curves.npz',**arrays)
        row['test_scale_curves_sha256']=file_sha(path.parent/'test_scale_curves.npz')
    write_json(path,row)
    with pytest.raises(ValueError):verify_measurements(path)


def test_failed_reference_and_dependent_are_verified_as_missing_measurements(tmp_path):
    ref,base=make_receipt(tmp_path/'ref',known=False)
    dependent,row=make_receipt(tmp_path/'dependent',known=False,reference=ref)
    assert base['measurement_status']==row['measurement_status']=='selection_failed'
    assert row['metrics']=={} and row['selected_lambda'] is None
    assert row['selection']['reference_failure_reason']=='no_knee'
    (dependent.parent/'reference_complete.json').write_text('{}')
    with pytest.raises(ValueError,match='reference receipt changed'):verify_measurements(dependent)


def test_nonfinite_json_is_not_exported_as_a_success(tmp_path):
    with pytest.raises(ValueError):write_json(tmp_path/'invalid.json',dict(mae=float('nan')))
    assert not (tmp_path/'invalid.json').exists()


def test_primary_supervised_and_kneedle_share_every_candidate_including_the_other_tail():
    grid=m.common_scales();target=np.ones(4)
    error=np.full(29,2.);error[14]=1.;error[-2]=.1
    curve=target[:,None]+error[None,:]
    scale,receipt=select_scale(curve,grid,target,SimpleNamespace(target_policy='known_lid'))
    assert scale==grid[-2]  # The old one-sided rule stopped at lambda0.5.
    plan=m.known_plan(curve,grid,target)
    assert receipt['candidate_lambdas']==plan['common_scales']==grid.tolist()
    assert receipt==plan['supervised_common_grid']
    assert plan['primary_automatic_metric']=='kneedle_common_grid'
    assert plan['legacy_role']=='historical_reproduction_only'
    assert receipt['status']=='selected'


def test_full_grid_boundary_ties_keep_smallest_scale_and_are_reported():
    grid=m.common_scales();curve=np.full((4,29),3.)
    curve[:,[0,-1]]=1.
    scale,receipt=select_scale(curve,grid,np.ones(4),SimpleNamespace(target_policy='known_lid'))
    assert scale==1/256 and receipt['scale_unresolved'] is True
    assert receipt['status']=='scale_unresolved'


def test_export_cannot_substitute_legacy_kneedle_for_primary_automatic_result(tmp_path):
    path,row=make_receipt(tmp_path/'known')
    row['automatic_metrics']=dict(row['diagnostic_metrics']['kneedle_legacy'])
    row['automatic_metrics']['mean']+=1.
    write_json(path,row)
    with pytest.raises(ValueError,match='primary automatic metrics'):verify_measurements(path)
