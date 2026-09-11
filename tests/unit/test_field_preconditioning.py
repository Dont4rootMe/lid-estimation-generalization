import yaml
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.training import TrainingConfig, TrainingResult, _canonical_family, _bottleneck_architecture_from_training_config, _vp_architecture_from_training_config, predict_lid
from models.preconditioned_field import build_bottleneck
from models.vp_baseline import ConditionedBottleneckMLP
from experiments.global_campaign_v2 import _prediction_curve

CONTRACTS = yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/lambda_repair/native_contracts.yaml').read_text())['model_contracts']
CONTRACTS = [c for c in CONTRACTS if c['variant_id']!='scale_conditioned_nf']


@pytest.mark.parametrize('contract',CONTRACTS,ids=[c['variant_id'] for c in CONTRACTS])
@pytest.mark.parametrize('mode,backbone',[
    ('gaussian_v1','bottleneck_v1'),('gaussian_no_input_skip_v1','bottleneck_v1'),
    ('covariance_span_v1','bottleneck_v1'),('covariance_span_v1','spectral_residual_v1')])
def test_all_native_readouts_match_gaussian(contract,mode,backbone):
    torch.set_num_threads(1)
    cfg=TrainingConfig.from_mapping(contract['model']['training'])
    family=_canonical_family(contract['model']['family'])
    kwargs={'vp_hidden_sizes':(24,16,8)} if family=='vp_diffusion' else {'field_hidden_sizes':(24,16,8)}
    rank=3 if mode=='covariance_span_v1' else 6
    cfg=replace(cfg,field_preconditioning=mode,field_backbone=backbone,
        field_projection_rank=rank if mode=='covariance_span_v1' else None,**kwargs)
    if family=='vp_diffusion':arch=_vp_architecture_from_training_config(cfg,ambient_dim=6)
    else:arch=_bottleneck_architecture_from_training_config(cfg,family=family,ambient_dim=6)
    model=build_bottleneck(arch,family,cfg).double()
    if mode=='covariance_span_v1':
        with torch.no_grad():model.basis[:3]=torch.eye(3,dtype=torch.float64)
    result=TrainingResult(family=family,model=model,config=cfg,history=(),best_epoch=0,best_validation_loss=0,
        checkpoint_path=Path('test-only'),checkpoint_sha256='',normalization_mean=torch.zeros(6),normalization_scale=1.,preprocessing={},preprocessing_sha256='')
    query=np.random.default_rng(3).normal(size=(4,6)).astype(np.float32)
    scales=np.array([1/256,1/16,.5,1.,4.,16.,64.])
    spec=copy.deepcopy(contract['model']);spec['derivative_backend']='exact';spec['trace_probes']=0
    measured=_prediction_curve(predict_lid,result,query,scales,model=spec,seed=0,batch_size=4,readout='full')
    expected=np.array([rank/(1+s*s)+s*s/(1+s*s)**2*(query[:,:rank].astype(float)**2).sum(1)
        +(query[:,rank:].astype(float)**2).sum(1)/s**2 for s in scales]).T
    np.testing.assert_allclose(measured,expected,atol=2e-9,rtol=2e-9)
    if mode=='covariance_span_v1':
        spec['derivative_backend']='active_exact'
        fast=_prediction_curve(predict_lid,result,query,scales,model=spec,seed=0,batch_size=4,readout='full')
        np.testing.assert_allclose(fast,measured,atol=2e-9,rtol=2e-9)


def test_legacy_builder_and_gradients_unchanged():
    cfg=TrainingConfig(field_hidden_sizes=(24,16,8),time_embedding_dim=128)
    arch=_bottleneck_architecture_from_training_config(cfg,family='gaussian_diffusion',ambient_dim=6)
    torch.manual_seed(25);old=ConditionedBottleneckMLP(6,(24,16,8),128,condition_transform='log').double()
    torch.manual_seed(25);new=build_bottleneck(arch,'gaussian_diffusion',cfg).double()
    x=torch.randn(8,6,dtype=torch.float64);c=torch.ones(8,dtype=torch.float64)
    torch.testing.assert_close(old(x,c),new(x,c),rtol=0,atol=0)
    old(x,c).square().sum().backward();new(x,c).square().sum().backward()
    for a,b in zip(old.parameters(),new.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_no_raw_skip_changes_derivative_at_multiple_scales():
    cfg=TrainingConfig(field_hidden_sizes=(24,16,8),time_embedding_dim=128,field_preconditioning='gaussian_no_input_skip_v1')
    arch=_bottleneck_architecture_from_training_config(cfg,family='gaussian_diffusion',ambient_dim=6)
    model=build_bottleneck(arch,'gaussian_diffusion',cfg).double()
    with torch.no_grad():model.decoder[-1].weight[:,:6].fill_(100.)
    x=torch.ones(1,6,dtype=torch.float64,requires_grad=True)
    for s in [.1,1.,64.]:
        y=model(x,torch.tensor([s],dtype=torch.float64))
        torch.testing.assert_close(y,x/(1+s*s))
    y.square().sum().backward()
    assert model.decoder[-1].weight.grad[:,:6].count_nonzero()==0


def test_covariance_span_gaussian_value_and_jacobian():
    cfg=TrainingConfig(field_hidden_sizes=(24,16,8),time_embedding_dim=128,
        field_preconditioning='covariance_span_v1',field_projection_rank=3)
    arch=_bottleneck_architecture_from_training_config(cfg,family='gaussian_diffusion',ambient_dim=6)
    model=build_bottleneck(arch,'gaussian_diffusion',cfg).double()
    with torch.no_grad():
        model.basis[:3,:]=torch.eye(3,dtype=torch.float64)
        model.variances.copy_(torch.tensor([.2,2.,20.],dtype=torch.float64))
    x=torch.randn(6,dtype=torch.float64)
    for s in [.00390625,.1,2.,64.]:
        gain=torch.cat((model.variances/(model.variances+s*s),torch.zeros(3,dtype=torch.float64)))
        value=model(x[None],s)[0]
        jac=torch.func.jacrev(lambda y:model(y[None],s)[0])(x)
        torch.testing.assert_close(value,gain*x,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(jac,torch.diag(gain),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('backbone',['bottleneck_v1','spectral_residual_v1'])
def test_covariance_fit_and_checkpoint_reload(tmp_path,backbone):
    from models.training import train_model,load_checkpoint
    rng=np.random.default_rng(94);x=np.zeros((96,6),dtype=np.float32);x[:,:3]=rng.normal(size=(96,3))
    cfg=TrainingConfig(field_hidden_sizes=(24,16,8),time_embedding_dim=128,
        field_preconditioning='covariance_span_v1',field_backbone=backbone,device='cpu',training_mode='fixed_steps_v1',
        steps=4,warmup_steps=0,validation_interval_steps=2,sigma_min=.01,sigma_max=64.,batch_size=16,early_stopping_patience=None)
    trained=train_model('diffusion',x[:80],x[80:],cfg,tmp_path/'model.pt')
    restored=load_checkpoint(tmp_path/'model.pt',device='cpu')
    assert trained.config.field_projection_rank==restored.config.field_projection_rank==3
    assert restored.model.omitted_variance_fraction<1e-12
    for key,val in trained.model.state_dict().items():torch.testing.assert_close(val,restored.model.state_dict()[key],rtol=0,atol=0)


@pytest.mark.parametrize('contract',CONTRACTS,ids=[c['variant_id'] for c in CONTRACTS])
def test_nonzero_data_scaled_residual_native_and_exact_span_trace(contract):
    from models.preconditioned_field import covariance_span_value_and_trace
    cfg=TrainingConfig.from_mapping(contract['model']['training'])
    family=_canonical_family(contract['model']['family'])
    kwargs={'vp_hidden_sizes':(24,16,8)} if family=='vp_diffusion' else {'field_hidden_sizes':(24,16,8)}
    cfg=replace(cfg,field_preconditioning='covariance_span_v1',field_projection_rank=3,
                field_residual_scaling='data_v1',**kwargs)
    arch=(_vp_architecture_from_training_config(cfg,ambient_dim=6) if family=='vp_diffusion'
          else _bottleneck_architecture_from_training_config(cfg,family=family,ambient_dim=6))
    model=build_bottleneck(arch,family,cfg).double()
    with torch.no_grad():
        model.basis[:3]=torch.eye(3,dtype=torch.float64)
        model.core.decoder[-1].weight.normal_(0,.01)
    result=TrainingResult(family=family,model=model,config=cfg,history=(),best_epoch=0,best_validation_loss=0,
        checkpoint_path=Path('test-only'),checkpoint_sha256='',normalization_mean=torch.zeros(6),
        normalization_scale=1.,preprocessing={},preprocessing_sha256='')
    query=np.random.default_rng(3).normal(size=(4,6)).astype(np.float32)
    spec=copy.deepcopy(contract['model']);spec['trace_probes']=0
    scales=np.array([1/256,.5,64.]);spec['derivative_backend']='exact'
    full=_prediction_curve(predict_lid,result,query,scales,model=spec,seed=0,batch_size=4,readout='full')
    spec['derivative_backend']='active_exact'
    active=_prediction_curve(predict_lid,result,query,scales,model=spec,seed=0,batch_size=4,readout='full')
    np.testing.assert_allclose(active,full,rtol=2e-9,atol=2e-9)


@pytest.mark.parametrize('backbone',['bottleneck_v1','spectral_residual_v1'])
def test_capacity_reports_actual_projected_model(backbone):
    from models.training import field_parameter_count
    cfg=TrainingConfig(field_hidden_sizes=(24,16,8),time_embedding_dim=128,
        field_preconditioning='covariance_span_v1',field_projection_rank=3,field_backbone=backbone)
    arch=_bottleneck_architecture_from_training_config(cfg,family='gaussian_diffusion',ambient_dim=30)
    model=build_bottleneck(arch,'gaussian_diffusion',cfg)
    assert field_parameter_count('diffusion',cfg,30)==sum(p.numel() for p in model.parameters())
    with pytest.raises(ValueError,match='covariance rank'):
        field_parameter_count('diffusion',replace(cfg,field_projection_rank=None),30)


def test_covariance_does_not_remove_genuine_weak_coordinate():
    from models.preconditioned_field import training_covariance_span
    x=torch.tensor(np.random.default_rng(72).normal(size=(512,3)),dtype=torch.float32)
    x[:,2]*=1e-5
    basis,variance,omitted=training_covariance_span(x,'cpu')
    assert basis.shape==(3,3)
    assert omitted==0


def test_small_constant_centering_residue_is_not_a_third_dimension():
    from models.preconditioned_field import training_covariance_span
    x=torch.tensor(np.random.default_rng(72).normal(size=(512,3)),dtype=torch.float32)
    x[:,:2]-=x[:,:2].mean(0);x[:,2]=3e-5
    basis,variance,omitted=training_covariance_span(x,'cpu')
    assert basis.shape==(3,2)


def test_gaussian_tail_controls_large_noise_residual_trace_order():
    from models.preconditioned_field import covariance_span_value_and_trace
    class IdentityCore(torch.nn.Module):
        def forward(self,x,condition):return x
    cfg=TrainingConfig(field_hidden_sizes=(24,16,8),time_embedding_dim=128,
        field_preconditioning='covariance_span_v1',field_projection_rank=3,
        field_residual_scaling='gaussian_tail_v1')
    arch=_bottleneck_architecture_from_training_config(cfg,family='gaussian_diffusion',ambient_dim=3)
    model=build_bottleneck(arch,'gaussian_diffusion',cfg).double();model.core=IdentityCore()
    with torch.no_grad():
        model.basis.copy_(torch.eye(3));model.variances.copy_(torch.tensor([1.,4.,9.]))
    x=torch.tensor([[.1,.2,.3]],dtype=torch.float64)
    _,r=covariance_span_value_and_trace(model,x,1e4)
    # Gaussian gain and the nonzero residual each contribute sum(v)/lambda².
    torch.testing.assert_close(r*1e8,torch.tensor([28.],dtype=torch.float64),rtol=2e-7,atol=0)


@pytest.mark.parametrize('contract',[c for c in CONTRACTS if c['model']['family']=='independent_affine_flow'],
                         ids=[c['variant_id'] for c in CONTRACTS if c['model']['family']=='independent_affine_flow'])
def test_fast_affine_channel_preserves_loss_and_every_gradient(contract):
    from models.affine_flow import affine_schedule_state
    from models import training
    cfg=TrainingConfig.from_mapping(contract['model']['training'])
    cfg=replace(cfg,field_hidden_sizes=(24,16,8),field_preconditioning='covariance_span_v1',
                field_projection_rank=3,noise_pairing='antithetic_v1')
    arch=_bottleneck_architecture_from_training_config(cfg,family='independent_affine_flow',ambient_dim=6)
    model=build_bottleneck(arch,'independent_affine_flow',cfg).double()
    model._lid_noise_pairing='antithetic_v1'
    with torch.no_grad():
        model.basis[:3]=torch.eye(3,dtype=torch.float64)
        model.core.decoder[-1].weight.normal_(0,.01)
    old=copy.deepcopy(model)
    old.channel=lambda condition:(condition.exp(),affine_schedule_state(condition.exp(),cfg.flow_schedule).alpha)
    x=torch.tensor(np.random.default_rng(24).normal(size=(12,6)),dtype=torch.float64)
    a=training._objective('independent_affine_flow',model,x,cfg,torch.Generator().manual_seed(20))
    b=training._objective('independent_affine_flow',old,x,cfg,torch.Generator().manual_seed(20))
    torch.testing.assert_close(a,b,rtol=0,atol=0);a.backward();b.backward()
    for p,q in zip(model.parameters(),old.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
