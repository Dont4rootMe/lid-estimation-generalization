import yaml
from dataclasses import replace
from pathlib import Path
import copy,json
import numpy as np
import pytest
import torch
from torch import nn
from models.empirical_target import EmpiricalPosteriorTarget,denoising_target
from models import training


def test_empirical_conditional_mean_and_loss_gradient_identity():
    torch.set_num_threads(1)
    bank=torch.tensor([[-1.,0.],[.5,1.],[2.,-.5]],dtype=torch.float64)
    y=torch.tensor([[.2,-.1],[.8,.3]],dtype=torch.float64);scale=torch.tensor([.7,1.2],dtype=torch.float64)
    teacher=EmpiricalPosteriorTarget(bank);mean=teacher(y,scale)
    w=torch.softmax(-((y[:,None]-bank[None]).square().sum(-1))/(2*scale[:,None]**2),dim=1)
    torch.testing.assert_close(mean,w@bank,rtol=1e-14,atol=1e-14)
    q=torch.tensor([[.3,.8],[-.4,.1]],dtype=torch.float64,requires_grad=True)
    original=(w*((q[:,None]-bank[None]).square().sum(-1))/scale[:,None]**2).sum()
    rb=((q-mean).square().sum(-1)/scale**2).sum()
    a=torch.autograd.grad(original,q)[0];b=torch.autograd.grad(rb,q)[0]
    torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
    assert copy.deepcopy(teacher) is teacher


def test_teacher_zero_noise_and_validation_bypass():
    x=torch.tensor([[100.,-20.],[1.,2.]],dtype=torch.float64)
    teacher=EmpiricalPosteriorTarget(x)
    torch.testing.assert_close(teacher(x,torch.zeros(2)),x,rtol=0,atol=0)
    model=nn.Linear(2,2);model._lid_use_empirical_target=True;model._lid_empirical_teacher=teacher
    model.eval();fallback=torch.ones_like(x)
    assert denoising_target(model,x,1.,fallback) is fallback


CONTRACTS=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/lambda_repair/native_contracts.yaml').read_text())['model_contracts']
CONTRACTS=[c for c in CONTRACTS if c['variant_id']!='scale_conditioned_nf']


@pytest.mark.parametrize('contract',CONTRACTS,ids=[c['variant_id'] for c in CONTRACTS])
def test_every_native_family_empirical_target_is_finite_and_validation_unchanged(contract):
    from models.preconditioned_field import build_bottleneck
    cfg=training.TrainingConfig.from_mapping(contract['model']['training']);family=training._canonical_family(contract['model']['family'])
    kwargs={'vp_hidden_sizes':(24,16,8)} if family=='vp_diffusion' else {'field_hidden_sizes':(24,16,8)}
    cfg=replace(cfg,field_preconditioning='covariance_span_v1',field_projection_rank=3,**kwargs)
    arch=(training._vp_architecture_from_training_config(cfg,ambient_dim=6) if family=='vp_diffusion'
          else training._bottleneck_architecture_from_training_config(cfg,family=family,ambient_dim=6))
    model=build_bottleneck(arch,family,cfg).double()
    with torch.no_grad():model.basis[:3]=torch.eye(3,dtype=torch.float64)
    bank=torch.tensor(np.random.default_rng(31).normal(size=(32,6)),dtype=torch.float64)
    model._lid_empirical_teacher=EmpiricalPosteriorTarget(bank);model._lid_use_empirical_target=True
    model.train();loss=training._objective(family,model,bank[:8],cfg,torch.Generator().manual_seed(74))
    assert torch.isfinite(loss);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval();a=training._objective(family,model,bank[:8],cfg,torch.Generator().manual_seed(75))
    model._lid_use_empirical_target=False;b=training._objective(family,model,bank[:8],cfg,torch.Generator().manual_seed(75))
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['gaussian_v1','gaussian_no_input_skip_v1','covariance_span_v1'])
def test_preconditioned_vp_handles_rng_zero_time(mode):
    from models.preconditioned_field import build_bottleneck
    from models.vp_baseline import vp_loss,VPSchedule
    cfg=training.TrainingConfig(vp_hidden_sizes=(24,16,8),time_embedding_dim=128,vp_beta_min=.1,vp_beta_max=20.,
        field_preconditioning=mode,field_projection_rank=3 if mode=='covariance_span_v1' else None)
    arch=training._vp_architecture_from_training_config(cfg,ambient_dim=6)
    model=build_bottleneck(arch,'vp_diffusion',cfg).float()
    if mode=='covariance_span_v1':
        with torch.no_grad():model.basis[:3]=torch.eye(3)
    x=torch.randn(8,6);t=torch.zeros(8);noise=torch.randn_like(x)
    torch.testing.assert_close(model(x,t),torch.zeros_like(x),rtol=0,atol=0)
    loss=vp_loss(model,x,t,noise,VPSchedule(.1,20.));loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
