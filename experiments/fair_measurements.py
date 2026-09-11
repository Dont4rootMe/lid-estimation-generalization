"""Replayable measurement rules; no model fitting or test-label scale selection.

The legacy FLIPD grid remains a distinct baseline. The common-grid diagnostic
changes only the candidate grid; it does not promise that a knee estimates LID.
Its supervised comparator is frozen on the source-train holdout on that grid.
"""
import importlib.metadata
import warnings

import numpy as np
from kneed import KneeLocator

from experiments import global_campaign_v2 as v2


def validate_runtime():
    if importlib.metadata.version('kneed') != v2.PINNED_KNEED_VERSION:
        raise ValueError('measurement requires kneed '+v2.PINNED_KNEED_VERSION)


def common_scales():
    return 2.**(np.arange(-16,13)/2)


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


def validate_selection(selected,selection,*,reference=False):
    status=selection.get('status')
    if status=='selection_failed':
        if selected is not None or not selection.get('failure_reason'):
            raise ValueError('failed selection requires no scale and an explicit failure reason')
    elif status in ('selected','scale_unresolved'):
        if selected is None or isinstance(selected,bool) or not np.isfinite(selected):
            raise ValueError('successful selection requires a finite scale')
        grid=v2.unknown_reference_lambdas() if reference else common_scales()
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


def known_plan(curve,scales,target):
    values,grid=validate_curve(curve,scales,target)
    require_grid(grid,common_scales())
    if target is None:raise ValueError('known-LID holdout targets are required')
    errors=np.abs(values-np.asarray(target)[:,None]).mean(axis=0)
    index=int(np.flatnonzero(errors<=errors.min()+1e-12)[0])
    return dict(protocol='known_lid_measurements_v2',kneed_version=v2.PINNED_KNEED_VERSION,
        coordinate='reference_vp_time',S=1.0,curve='convex',direction='decreasing',
        online=False,interp_method='interp1d',fallback='ambient_dimension',
        common_scales=grid.tolist(),legacy_scales=legacy_scales().tolist(),
        supervised_common_grid=dict(selected_index=index,selected_lambda=float(grid[index]),
            criterion='minimum_source_train_holdout_mae_on_all_29_candidates',
            tie_rule='smallest_lambda_within_1e-12',holdout_mae=float(errors[index])),
        test_labels_used_for_selection=False)


def pointwise(curve,scales,ambient_dim):
    values,grid=validate_curve(curve,scales)
    if not np.isfinite(ambient_dim) or ambient_dim<=0:raise ValueError('invalid ambient dimension')
    times=v2.flipd_timesteps() if np.array_equal(grid,legacy_scales()) else v2.vp_time_from_lambda(grid)
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


def known_results(common_curve,legacy_curve,target,plan,ambient_dim):
    common,grid=validate_curve(common_curve,plan['common_scales'],target)
    legacy,old_grid=validate_curve(legacy_curve,plan['legacy_scales'],target)
    require_grid(grid,common_scales());require_grid(old_grid,legacy_scales())
    if len(common)!=len(legacy) or target is None:raise ValueError('known-LID query sets differ')
    selected=plan['supervised_common_grid']
    index=selected['selected_index']
    if not 0<=index<len(grid) or grid[index]!=selected['selected_lambda']:
        raise ValueError('invalid frozen common-grid supervised scale')
    results=dict(supervised_common_grid=dict(metric(common[:,index],target),
        selected_lambda=float(grid[index]),selection_partition='source_train_holdout'))
    arrays=dict(common_scales=grid,common_prediction=common,
        legacy_scales=old_grid,legacy_prediction=legacy)
    for name,values,scales in [('kneedle_legacy',legacy,old_grid),('kneedle_common_grid',common,grid)]:
        prediction,indices,fallback=pointwise(values,scales,ambient_dim)
        results[name]=dict(metric(prediction,target),fallback_n=int(fallback.sum()),
            detected_n=int((~fallback).sum()),
            boundary_n=int(((indices==0)|(indices==len(scales)-1)).sum()),
            selected_lambda_counts=[dict(index=int(j),selected_lambda=float(scales[j]),
                n=int((indices==j).sum())) for j in np.unique(indices) if j>=0])
        arrays[name+'_prediction']=prediction
        arrays[name+'_index']=indices
        arrays[name+'_fallback']=fallback
    return results,arrays
