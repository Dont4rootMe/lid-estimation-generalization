"""Model-window and selection regression checks; no optimizer steps."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from kneed import KneeLocator

from experiments import fair_measurements as m,global_campaign_v2 as v2
from experiments.fair_campaign import prediction_spec,select_scale,verify_measurements
from experiments.fair_protocol import native_contracts
from models.native_tasks import valid_scales


def config(variant):
    return SimpleNamespace(**native_contracts()[variant]['training'])


WINDOWS={
    'vp_diffusion':(1/256,math.sqrt(math.expm1(10.05))),
    've_diffusion':(.01,50.),
    'rectified_flow':(1/256,999.),
    'schrodinger_bridge':(math.sqrt(.02),math.sqrt(.4)),
    **{v:(1/256,64.) for v in ('scale_conditioned_nf','posterior_rectified_flow',
        'direct_log_noise_affine_flow','posterior_log_noise_affine_flow',
        'direct_vp_trigonometric_flow','posterior_vp_trigonometric_flow',
        't_flowmatching','pfgmpp')},
}


@pytest.mark.parametrize('variant',WINDOWS)
def test_all_methods_get_29_geometric_in_support_candidates(variant):
    cfg=config(variant);grid=m.model_scales(cfg)
    assert len(grid)==29 and np.all(np.diff(grid)>0)
    np.testing.assert_allclose(grid[[0,-1]],WINDOWS[variant],rtol=1e-14,atol=0)
    gaps=np.diff(np.log(grid))
    np.testing.assert_allclose(gaps,gaps[0],rtol=1e-12,atol=0)
    assert valid_scales(cfg,grid,nf_stencil=variant=='scale_conditioned_nf').all()
    if WINDOWS[variant]==(1/256,64.):
        np.testing.assert_array_equal(grid,m.common_scales())


def test_nf_every_ols5_neighbor_remains_in_the_training_interval():
    cfg=config('scale_conditioned_nf')
    stencil=m.model_scales(cfg)[:,None]*np.exp(.05*np.arange(-2,3))[None,:]
    assert stencil.min()>=cfg.epsilon_min and stencil.max()<=cfg.epsilon_max


@pytest.mark.parametrize('variant',['vp_diffusion','ve_diffusion','rectified_flow','schrodinger_bridge'])
def test_prediction_dispatch_accepts_new_endpoints_as_physical_lambda(variant):
    cfg=config(variant);grid=m.model_scales(cfg)
    spec=prediction_spec(variant,{'kind':'vector'},cfg)
    calls=[]
    def predictor(model,query,scale,**kwargs):
        calls.append(scale)
        return np.full(len(query),scale)
    curve=v2._prediction_curve(predictor,None,np.zeros((2,3)),grid,model=spec,
        seed=0,batch_size=2,readout='full')
    np.testing.assert_array_equal(calls,grid)
    np.testing.assert_array_equal(curve,np.tile(grid,(2,1)))


@pytest.mark.parametrize('variant',['vp_diffusion','ve_diffusion','rectified_flow','schrodinger_bridge'])
def test_new_endpoints_work_through_actual_native_readout_without_training(variant):
    from dataclasses import replace
    from experiments.fair_protocol import resolve,geometry,build_model
    from models import training
    geo=geometry('fixture','coefficients',(3,))
    cfg,_=resolve(variant,geo,device='cpu',steps=2,preflight=True)
    cfg=replace(cfg,field_residual_width=16)
    model=build_model(variant,cfg,3)
    grid=m.model_scales(cfg)[[0,-1]]
    values=v2._prediction_curve(training.predict_lid,model,np.zeros((1,3)),grid,
        model=prediction_spec(variant,geo,cfg),seed=0,batch_size=1,readout='full')
    # The untrained native cores have zero outputs: q=x except RF, where q=t*x.
    expected=3/(1+grid) if variant=='rectified_flow' else np.full(2,3.)
    np.testing.assert_allclose(values[0],expected,rtol=1e-6,atol=1e-7)


def test_primary_kneedle_uses_log_lambda_beyond_vp_support():
    grid=m.model_scales(config('rectified_flow'))
    row=np.exp(-6*np.linspace(0,1,29))
    oracle=KneeLocator(np.log(grid),row,S=1.,curve='convex',direction='decreasing',
        online=False,interp_method='interp1d')
    assert oracle.knee is not None
    expected=int(np.argmin(abs(np.log(grid)-oracle.knee)))
    value,index,fallback=m.pointwise(row[None,:],grid,30,coordinate='log_lambda')
    assert index[0]==expected and value[0]==row[expected] and not fallback[0]
    with pytest.raises(v2.GlobalCampaignError):v2.vp_time_from_lambda(grid)


def test_new_grid_is_frozen_for_supervised_and_automatic_selection():
    cfg=config('schrodinger_bridge');grid=m.model_scales(cfg)
    curve=np.tile(1+np.arange(29,dtype=float)**2,(3,1));target=np.ones(3)
    cell=SimpleNamespace(target_policy='known_lid')
    chosen,selection=select_scale(curve,grid,target,cell,config=cfg)
    plan=m.known_plan(curve,grid,target,expected_grid=grid)
    assert chosen==grid[0] and selection['status']=='scale_unresolved'
    assert selection==plan['supervised_common_grid']
    assert plan['common_scales']==grid.tolist() and plan['coordinate']=='log_lambda'
    with pytest.raises(ValueError,match='frozen protocol'):
        select_scale(curve[:,:-1],grid[:-1],target,cell,config=cfg)


def test_reference_log_grid_choice_transfers_without_retuning():
    cfg=config('rectified_flow');grid=m.model_scales(cfg)
    curve=np.tile(np.exp(-6*np.linspace(0,1,29)),(3,1))
    source=SimpleNamespace(target_policy='sample_size',dataset='base',reference_dataset='base')
    chosen,selection=select_scale(curve,grid,None,source,config=cfg)
    assert chosen is not None and selection['coordinate']=='log_lambda'
    child=SimpleNamespace(target_policy='sample_size',dataset='child',reference_dataset='base')
    inherited,receipt=select_scale(-curve,grid,None,child,
        dict(selected_lambda=chosen,selection=selection),config=cfg)
    assert inherited==chosen and receipt['candidate_lambdas']==grid.tolist()
    assert receipt['criterion']=='reuse_reference_mean_kneedle'


@pytest.mark.parametrize('variant',['vp_diffusion','ve_diffusion','rectified_flow','schrodinger_bridge'])
def test_native_known_receipts_replay_with_the_actual_model_grid(tmp_path,fixture_input_manifest,variant):
    from tests.unit.test_fair_measurements import make_receipt
    path,row=make_receipt(tmp_path/'native',native=True,variant=variant)
    assert verify_measurements(path)['scale_protocol']==m.SCALE_PROTOCOL
    assert row['measurement_plan']['coordinate']=='log_lambda'


@pytest.mark.parametrize('success',[False,True])
def test_native_unknown_reference_receipts_replay_and_transfer(tmp_path,fixture_input_manifest,success):
    from tests.unit.test_fair_measurements import make_receipt
    path,base=make_receipt(tmp_path/'base',known=False,native=True,
        variant='rectified_flow',unknown_success=success)
    child,row=make_receipt(tmp_path/'child',known=False,native=True,
        variant='rectified_flow',reference=path)
    assert verify_measurements(child)['selected_lambda']==base['selected_lambda']
    assert (row['measurement_status']=='selected')==success
