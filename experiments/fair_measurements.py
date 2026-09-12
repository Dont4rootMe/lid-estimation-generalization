"""Replayable measurement rules; no model fitting or test-label scale selection.

Native v10 selectors share 29 model-specific geometric candidates and use
log(lambda) for Kneedle. The supervised scale is frozen on source-train holdout.
Legacy helper defaults and the narrow FLIPD diagnostic preserve old receipts.
"""
import importlib.metadata
import warnings

import numpy as np
from kneed import KneeLocator

from experiments import global_campaign_v2 as v2

PRIMARY_AUTOMATIC_METRIC='kneedle_common_grid'
SCALE_PROTOCOL='per_model_logspace_v1'
SUPERVISED_PROTOCOL='held_out_source_train_supervised_mae_model_logspace_v5'
POINTWISE_PROTOCOL='pointwise_kneedle_model_logspace_v4'
REFERENCE_PROTOCOL='held_out_reference_mean_kneedle_logspace_v3'


def validate_runtime():
    if importlib.metadata.version('kneed') != v2.PINNED_KNEED_VERSION:
        raise ValueError('measurement requires kneed '+v2.PINNED_KNEED_VERSION)


def common_scales():
    return 2.**(np.arange(-16,13)/2)


def model_scale_bounds(config):
    """Finite evaluation window; preserve native training and NF OLS5 support."""
    from models.native_tasks import scale_support
    lo,hi=scale_support(config)
    if lo==0:lo=1/256
    if np.isposinf(hi):hi=64.
    if config.native_variant=='scale_conditioned_nf':
        lo*=np.exp(.1);hi/=np.exp(.1)
    if not np.isfinite([lo,hi]).all() or not 0<lo<hi:
        raise ValueError('invalid model scale window')
    return float(lo),float(hi)


def model_scales(config):
    """29 geometric candidates over each model's full finite evaluation window."""
    from models.native_tasks import valid_scales
    lo,hi=model_scale_bounds(config)
    # Preserve the old half-octaves exactly for unchanged [1/256,64] windows.
    grid=common_scales() if (lo,hi)==(1/256,64.) else np.geomspace(lo,hi,29)
    grid[0],grid[-1]=lo,hi
    nf=config.native_variant=='scale_conditioned_nf'
    # OLS5's multiply/divide round trip may put an endpoint one ulp outside.
    for index,toward in ((0,hi),(-1,lo)):
        for _ in range(4):
            if valid_scales(config,[grid[index]],nf_stencil=nf)[0]:break
            grid[index]=np.nextafter(grid[index],toward)
    if not valid_scales(config,grid,nf_stencil=nf).all():
        raise ValueError('model grid exceeds native or NF stencil support')
    return grid


def legacy_scales():
    return v2.vp_lambda_from_time(v2.flipd_timesteps())


def validate_curve(curve,scales,target=None):
    values=np.asarray(curve,dtype=np.float64)
    grid=np.asarray(scales,dtype=np.float64)
    if (grid.ndim!=1 or len(grid)<3 or not np.isfinite(grid).all()
            or np.any(grid<=0) or np.any(np.diff(grid)<=0)):
        raise ValueError('scales must be finite, positive and strictly increasing')
    if (values.ndim!=2 or values.shape[0]==0 or values.shape[1]!=len(grid)
            or not np.isfinite(values).all()):
        raise ValueError('prediction curve must be finite nonempty N x scales')
    if target is not None:
        y=np.asarray(target)
        if y.shape!=(len(values),) or not np.isfinite(y).all():
            raise ValueError('targets must be finite and match the query count')
    return values,grid


def require_grid(actual,expected):
    if np.shape(actual)!=np.shape(expected) or not np.allclose(actual,expected,rtol=1e-12,atol=1e-12):
        raise ValueError('measurement grid differs from the frozen protocol')


def require_subgrid(actual,expected):
    actual=np.asarray(actual);expected=np.asarray(expected)
    if (actual.ndim!=1 or len(actual)<3 or np.any(np.diff(actual)<=0)
            or not np.isclose(actual[:,None],expected[None,:],rtol=1e-12,atol=1e-12).any(1).all()):
        raise ValueError('candidate grid is not an ordered supported subset of the master grid')


def validate_selection(selected,selection,*,reference=False,scales=None):
    status=selection.get('status')
    if status=='selection_failed':
        if selected is not None or not selection.get('failure_reason'):
            raise ValueError('failed selection requires no scale and an explicit failure reason')
    elif status in ('selected','scale_unresolved'):
        if selected is None or isinstance(selected,bool) or not np.isfinite(selected):
            raise ValueError('successful selection requires a finite scale')
        grid=np.asarray(scales) if scales is not None else (v2.unknown_reference_lambdas() if reference else common_scales())
        if not np.isclose(grid,selected,rtol=1e-12,atol=1e-12).any():
            raise ValueError('selected scale is outside the declared candidate grid')
        if reference and status!='selected':
            raise ValueError('unknown reference cannot have a supervised boundary status')
    else:
        raise ValueError('unrecognized selection status')
    return status


def metric(values,target,dtype='float32'):
    values=np.asarray(values)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('predictions must be finite and nonempty')
    result=dict(mean=float(values.mean()),n=len(values),dtype=dtype)
    if target is not None:
        target=np.asarray(target)
        if target.shape!=values.shape or not np.isfinite(target).all():
            raise ValueError('targets must be finite and match predictions')
        result['mae']=float(np.abs(values-target).mean())
    return result


def supervised_common_selection(curve,scales,target,*,expected_grid=None):
    values,grid=validate_curve(curve,scales,target)
    if expected_grid is None:require_subgrid(grid,common_scales())
    else:require_grid(grid,expected_grid)
    if target is None:raise ValueError('known-LID holdout targets are required')
    errors=np.abs(values-np.asarray(target)[:,None]).mean(axis=0)
    index=int(np.flatnonzero(errors<=errors.min()+1e-12)[0])
    unresolved=index in (0,len(grid)-1)
    return dict(protocol=SUPERVISED_PROTOCOL if expected_grid is not None else 'held_out_source_train_supervised_mae_native_support_v4',
        selection_quantity='full_density_lid',
        selected_index=index,selected_lambda=float(grid[index]),
        criterion=('minimum_source_train_holdout_mae_on_model_logspace' if expected_grid is not None
                   else 'minimum_source_train_holdout_mae_on_supported_master_grid_candidates'),
        master_lambdas=(np.asarray(expected_grid) if expected_grid is not None else common_scales()).tolist(),
        candidate_lambdas=grid.tolist(),candidate_mae=errors.tolist(),
        tie_rule='smallest_lambda_within_1e-12',holdout_mae=float(errors[index]),
        status='scale_unresolved' if unresolved else 'selected',scale_unresolved=unresolved)


def known_plan(curve,scales,target,*,legacy_grid=None,expected_grid=None):
    selection=supervised_common_selection(curve,scales,target,expected_grid=expected_grid)
    old_grid=legacy_scales() if legacy_grid is None else np.asarray(legacy_grid)
    require_subgrid(old_grid,legacy_scales())
    return dict(protocol='known_lid_measurements_v5' if expected_grid is not None else 'known_lid_measurements_v4',kneed_version=v2.PINNED_KNEED_VERSION,
        selection_quantity='full_density_lid',secondary_scale_rule='reuse_full_indices',
        coordinate='log_lambda' if expected_grid is not None else 'reference_vp_time',S=1.0,curve='convex',direction='decreasing',
        online=False,interp_method='interp1d',fallback='ambient_dimension',
        common_scales=selection['candidate_lambdas'],legacy_scales=old_grid.tolist(),
        primary_automatic_metric=PRIMARY_AUTOMATIC_METRIC,
        legacy_role=('historical_reproduction_only' if np.array_equal(old_grid,legacy_scales())
                     else 'native_support_filtered_legacy_grid'),supervised_common_grid=selection,
        test_labels_used_for_selection=False)


def pointwise(curve,scales,ambient_dim,*,coordinate='reference_vp_time'):
    values,grid=validate_curve(curve,scales)
    if not np.isfinite(ambient_dim) or ambient_dim<=0:raise ValueError('invalid ambient dimension')
    if coordinate=='log_lambda':times=np.log(grid)
    elif coordinate=='reference_vp_time':
        times=v2.flipd_timesteps() if np.array_equal(grid,legacy_scales()) else v2.vp_time_from_lambda(grid)
    else:raise ValueError('unknown Kneedle coordinate')
    index=np.full(len(values),-1,dtype=np.int64)
    prediction=np.full(len(values),float(ambient_dim))
    for i,row in enumerate(values):
        # Constant curves have no knee. Avoid kneed's divide-by-zero warning;
        # this is exactly its legacy ambient-dimension fallback.
        if np.ptp(row)==0:continue
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            locator=KneeLocator(times,row,S=1.,curve='convex',direction='decreasing',
                online=False,interp_method='interp1d')
        if locator.knee is not None:
            index[i]=int(np.argmin(abs(times-float(locator.knee))))
            prediction[i]=row[index[i]]
    return prediction,index,index<0


def reference_selection(curve,scales):
    """Existing reference-mean/descending-segment rule, on log(lambda)."""
    values,grid=validate_curve(curve,scales)
    mean=values.mean(0);maximum=int(np.argmax(mean));segment=mean[maximum:]
    receipt=dict(protocol=REFERENCE_PROTOCOL,coordinate='log_lambda',
        criterion='reference_mean_kneedle_after_global_maximum',
        candidate_lambdas=grid.tolist(),global_maximum_index=maximum,
        segment_size=len(segment),knee_log_lambda=None,selected_index=None,
        selected_lambda=None,status='selection_failed')
    if len(segment)<3:return None,dict(receipt,failure_reason='remaining_segment_too_short')
    if np.ptp(segment)==0:return None,dict(receipt,failure_reason='no_knee')
    coordinates=np.log(grid[maximum:])
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        locator=KneeLocator(coordinates,segment,S=1.,curve='convex',direction='decreasing',
            online=False,interp_method='interp1d')
    if locator.knee is None:return None,dict(receipt,failure_reason='no_knee')
    receipt['knee_log_lambda']=float(locator.knee)
    local=int(np.argmin(abs(coordinates-float(locator.knee))))
    if local in (0,len(segment)-1):return None,dict(receipt,failure_reason='knee_at_boundary')
    index=maximum+local
    return index,dict(receipt,status='selected',selected_index=index,selected_lambda=float(grid[index]))


def known_results(common_curve,legacy_curve,target,plan,ambient_dim,*,response_curves=None):
    if plan['primary_automatic_metric']!=PRIMARY_AUTOMATIC_METRIC:
        raise ValueError('primary automatic selection must use the common grid')
    common,grid=validate_curve(common_curve,plan['common_scales'],target)
    legacy,old_grid=validate_curve(legacy_curve,plan['legacy_scales'],target)
    if plan['protocol']=='known_lid_measurements_v4':require_subgrid(grid,common_scales())
    elif plan['protocol']!='known_lid_measurements_v5' or plan['coordinate']!='log_lambda':
        raise ValueError('unknown measurement plan or Kneedle coordinate')
    require_subgrid(old_grid,legacy_scales())
    if len(common)!=len(legacy) or target is None:raise ValueError('known-LID query sets differ')
    selected=plan['supervised_common_grid']
    index=selected['selected_index']
    if not 0<=index<len(grid) or grid[index]!=selected['selected_lambda']:
        raise ValueError('invalid frozen common-grid supervised scale')
    results=dict(supervised_common_grid=dict(metric(common[:,index],target),
        selected_lambda=float(grid[index]),selection_partition='source_train_holdout'))
    arrays=dict(common_scales=grid,common_prediction=common,
        legacy_scales=old_grid,legacy_prediction=legacy)
    if response_curves is not None:
        response_common,_=validate_curve(response_curves[0],grid,target)
        response_legacy,_=validate_curve(response_curves[1],old_grid,target)
        if response_common.shape!=common.shape or response_legacy.shape!=legacy.shape:
            raise ValueError('response and Full query sets differ')
        arrays.update(common_response=response_common,legacy_response=response_legacy)
        results['supervised_common_grid']['response']=metric(response_common[:,index],target)
    for name,values,scales in [('kneedle_legacy',legacy,old_grid),('kneedle_common_grid',common,grid)]:
        coordinate='reference_vp_time' if name=='kneedle_legacy' else plan['coordinate']
        prediction,indices,fallback=pointwise(values,scales,ambient_dim,coordinate=coordinate)
        results[name]=dict(metric(prediction,target),fallback_n=int(fallback.sum()),
            detected_n=int((~fallback).sum()),
            boundary_n=int(((indices==0)|(indices==len(scales)-1)).sum()),
            selected_lambda_counts=[dict(index=int(j),selected_lambda=float(scales[j]),
                n=int((indices==j).sum())) for j in np.unique(indices) if j>=0])
        arrays[name+'_prediction']=prediction
        arrays[name+'_index']=indices
        arrays[name+'_fallback']=fallback
        if response_curves is not None:
            response_curve=response_legacy if name=='kneedle_legacy' else response_common
            # A missing Full knee means no lambda exists for either readout.
            # Keep the identical ambient fallback and all queries in both MAEs.
            response=np.full(len(values),float(ambient_dim))
            valid=np.flatnonzero(~fallback)
            response[valid]=response_curve[valid,indices[valid]]
            arrays[name+'_response_prediction']=response
            results[name]['response']=metric(response,target)
    return results,arrays
