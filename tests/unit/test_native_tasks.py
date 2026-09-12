"""Equation/routing checks only: no optimizer step and no learning experiment."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from torch import nn

from experiments.fair_protocol import native_contracts, resolve, build_model, geometry
from models import training
from models.native_tasks import (VARIANTS, sample_batch, objective, NativeField,
                                valid_scales, pfgm_rms_factor, PosteriorReadout)


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(variant,image=False):
    geo=geometry('fixture','dataset' if image else 'coefficients',(8,8,1) if image else (3,))
    cfg,_=resolve(variant,geo,device='cpu',steps=2,preflight=True)
    if not image and variant!='scale_conditioned_nf':
        cfg=replace(cfg,field_residual_width=16)
    return cfg,geo


def test_roster_and_native_contracts():
    assert set(native_contracts())==set(VARIANTS)
    assert 'direct_rectified_flow' not in native_contracts()
    for variant in VARIANTS:
        cfg,_=config(variant)
        assert training.TrainingConfig.from_mapping(cfg.to_dict())==cfg
        family=training._canonical_family(native_contracts()[variant]['family'])
        assert training._model_contract(family,cfg)['native_variant']==variant


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('image',[False,True])
def test_native_objectives_and_serialized_architecture(variant,image):
    cfg,geo=config(variant,image)
    model=build_model(variant,cfg,geo['ambient_dim'])
    replica=build_model(variant,cfg,geo['ambient_dim'])
    replica.load_state_dict(model.state_dict(),strict=True)
    x=torch.randn(2,geo['ambient_dim'],generator=torch.Generator().manual_seed(9))
    family=training._canonical_family(native_contracts()[variant]['family'])
    model.train()
    loss=training._objective(family,model,x,cfg,torch.Generator().manual_seed(8))
    assert loss.ndim==0 and torch.isfinite(loss)
    loss.backward()  # Derivative smoke only; parameters are never updated.
    assert any(p.grad is not None for p in model.parameters())
    if variant=='schrodinger_bridge':
        assert all(p.grad is None for p in model.backward_core.parameters())


@pytest.mark.parametrize('variant',['vp_diffusion','ve_diffusion'])
def test_flipd_targets_share_the_corruption_noise(variant):
    cfg,_=config(variant)
    clean=torch.tensor([[1.,2.,3.],[-1.,4.,2.]])
    y,t,target,w=sample_batch(clean,cfg,torch.Generator().manual_seed(4))
    if variant=='vp_diffusion':
        from models.vp_baseline import VPSchedule
        a,b=VPSchedule(cfg.vp_beta_min,cfg.vp_beta_max).coefficients(t)
    else:
        a=torch.ones_like(t);b=cfg.sigma_min*(cfg.sigma_max/cfg.sigma_min)**t
    torch.testing.assert_close(y+b[:,None]*target,a[:,None]*clean)
    assert torch.equal(w,torch.ones_like(w))


def test_rf_and_interflow_targets_invert_to_clean_endpoint():
    clean=torch.randn(5,3)
    for variant in ('rectified_flow','direct_vp_trigonometric_flow'):
        cfg,_=config(variant)
        y,c,v,w=sample_batch(clean,cfg,torch.Generator().manual_seed(12))
        if variant=='rectified_flow':
            t=c/999;reconstructed=y+(1-t)[:,None]*v
        else:
            a=torch.sin(math.pi/2*c);b=torch.cos(math.pi/2*c)
            reconstructed=a[:,None]*y+(b*2/math.pi)[:,None]*v
        torch.testing.assert_close(reconstructed,clean)
        assert torch.equal(w,torch.ones_like(w))


def test_custom_log_noise_heads_really_have_different_raw_targets():
    clean=torch.randn(4,3)
    p,_=config('posterior_log_noise_affine_flow')
    d,_=config('direct_log_noise_affine_flow')
    yp,cp,tp,wp=sample_batch(clean,p,torch.Generator().manual_seed(1))
    yd,cd,td,wd=sample_batch(clean,d,torch.Generator().manual_seed(1))
    torch.testing.assert_close(yp,yd);torch.testing.assert_close(wp,wd)
    torch.testing.assert_close(tp,clean);torch.testing.assert_close(td,yd-clean)
    assert not torch.allclose(tp,td)


@pytest.mark.parametrize('variant',[v for v in VARIANTS if v!='scale_conditioned_nf'])
def test_pointwise_readout_routes_without_training(variant):
    cfg,geo=config(variant)
    model=build_model(variant,cfg,geo['ambient_dim'])
    x=np.array([[.1,.2,.3]],dtype=np.float32)
    # The DSB discretization supports lambda=.25; every other native recipe does too.
    response=training.predict_lid(model,x,.25,family=native_contracts()[variant]['family'],
        readout='response',divergence_backend='exact',trace_probes=0)
    assert response.shape==(1,) and np.isfinite(response).all()
    if variant=='pfgmpp':
        with pytest.raises(NotImplementedError,match='clipp'):
            training.predict_lid(model,x,.25,family='pfgmpp',readout='full',divergence_backend='exact')
    else:
        value=training.predict_lid(model,x,.25,family=native_contracts()[variant]['family'],
            readout='full',divergence_backend='exact',trace_probes=0)
        assert value.shape==(1,) and np.isfinite(value).all()


@pytest.mark.parametrize('image',[False,True])
def test_dsb_second_projection_and_native_scale_support(image):
    cfg,geo=config('schrodinger_bridge',image);model=build_model('schrodinger_bridge',cfg,geo['ambient_dim'])
    model.set_training_step(2)
    loss=objective(model,torch.zeros(2,geo['ambient_dim']),cfg,torch.Generator().manual_seed(0))
    loss.backward()
    assert all(p.grad is None for p in model.core.parameters())
    assert any(p.grad is not None for p in model.backward_core.parameters())
    assert valid_scales(cfg,[.25,1.,64.]).tolist()==[True,False,False]


def test_ve_masks_master_grid_without_changing_sampler():
    cfg,_=config('ve_diffusion')
    grid=2.**(np.arange(-16,13)/2)
    mask=valid_scales(cfg,grid)
    assert not mask[0] and not mask[-1]
    assert grid[mask][0]>=.01 and grid[mask][-1]<=50.


def test_pfgm_clipped_second_moment_is_dimension_dependent():
    assert pfgm_rms_factor(3,128) > 0
    # At ordinary image dimension the clipped tail mass is negligible, while
    # low ambient dimensions must not inherit the unclipped RMS conversion.
    assert pfgm_rms_factor(3,128)!=math.sqrt(128/126)
    assert math.isclose(pfgm_rms_factor(3072,128),math.sqrt(128/126),rel_tol=1e-5)


@pytest.mark.parametrize('variant',VARIANTS)
def test_portable_checkpoint_fixture_without_optimization(variant,tmp_path):
    cfg,geo=config(variant);model=build_model(variant,cfg,geo['ambient_dim'])
    if variant=='schrodinger_bridge':model.set_training_step(2)
    family=training._canonical_family(native_contracts()[variant]['family'])
    # Synthetic serialization metadata only; this is explicitly not a fit.
    losses=(1.,2.) if variant=='schrodinger_bridge' else (2.,1.)
    history=tuple(training.StepMetrics(step=i+1,examples_seen=(i+1)*cfg.batch_size,
        train_loss=loss,validation_loss=loss,learning_rate=cfg.learning_rate)
        for i,loss in enumerate(losses))
    mean=torch.zeros(geo['ambient_dim']);prep,sha=training._preprocessing_identity(mean,1.,normalized=cfg.normalize)
    path=tmp_path/'serialization_fixture.pt'
    training._save_fixed_step_checkpoint(path,family=family,model=model,config=cfg,
        history=history,best_step=2,best_validation_loss=losses[-1],best_state=model.state_dict(),
        final_state=model.state_dict(),initial_validation_loss=3.,normalization_mean=mean,
        normalization_scale=1.,preprocessing=prep,preprocessing_sha256=sha)
    restored=training.load_checkpoint(path,device='cpu')
    assert restored.config==cfg
    assert restored.model_contract==training._model_contract(family,cfg)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.model.state_dict()[k],rtol=0,atol=0)


@pytest.mark.parametrize('variant',VARIANTS)
def test_training_entry_reaches_optimizer_but_never_updates(variant,tmp_path,monkeypatch):
    cfg,geo=config(variant)
    cfg=replace(cfg,batch_size=2)
    class StopBeforeUpdate(Exception):pass
    def stop(*args,**kwargs):raise StopBeforeUpdate
    monkeypatch.setattr(torch.optim.AdamW,'step',stop)
    x=np.arange(12,dtype=np.float32).reshape(4,3)/10
    with pytest.raises(StopBeforeUpdate):
        training.train_model(native_contracts()[variant]['family'],x,x,cfg,tmp_path/'never_written.pt')
    assert not (tmp_path/'never_written.pt').exists()


def test_every_native_support_has_usable_selection_grids():
    from experiments.fair_campaign import supported_grid
    from experiments.fair_measurements import common_scales,legacy_scales,known_plan,known_results
    for variant in VARIANTS:
        cfg,_=config(variant)
        grid=supported_grid(cfg,common_scales());legacy=supported_grid(cfg,legacy_scales())
        curve=np.ones((2,len(grid)));old=np.ones((2,len(legacy)));target=np.ones(2)
        plan=known_plan(curve,grid,target,legacy_grid=legacy)
        metrics,arrays=known_results(curve,old,target,plan,3)
        assert arrays['common_scales'].tolist()==grid.tolist()
        assert metrics['kneedle_common_grid']['fallback_n']==2


def test_previously_collapsed_routes_have_distinct_finite_network_gradients():
    variants=('ve_diffusion','schrodinger_bridge','posterior_rectified_flow',
        'direct_log_noise_affine_flow','posterior_log_noise_affine_flow',
        'direct_vp_trigonometric_flow','posterior_vp_trigonometric_flow')
    gradients=[];state=None
    clean=torch.tensor([[.3,-.8,1.2],[1.1,.2,-.6]],dtype=torch.float64)
    for variant in variants:
        cfg,geo=config(variant);model=build_model(variant,cfg,3).double()
        if state is None:
            with torch.no_grad():model.core.output.weight.fill_(.01)
            state={k:v.clone() for k,v in model.core.state_dict().items()}
        else:model.core.load_state_dict(state)
        loss=objective(model,clean,cfg,torch.Generator().manual_seed(17))
        grad=torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,tuple(model.core.parameters()))])
        for previous in gradients:
            assert torch.linalg.vector_norm(grad-previous)>1e-8*max(float(grad.norm()),float(previous.norm()))
        gradients.append(grad)
