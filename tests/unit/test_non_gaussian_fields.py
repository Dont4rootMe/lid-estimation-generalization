"""Native-kernel, conditioning, trace and common-training validity gates."""
import math

import numpy as np
import pytest
from scipy.stats import betaprime, kstest, norm
import torch

from experiments.fair_protocol import resolve, geometry, build_model, native_contracts, parameter_count
from experiments.fair_outputs import readouts
from models import training
from models.non_gaussian_fields import (student_unit_rms, sample_rms_scale,
    conditional_rms_scale, loss)
from models.preconditioned_field import covariance_span_value_and_trace
from models.neural_fields import exact_divergence

torch.set_num_threads(2)


def vector_config(variant, ambient=7, rank=2):
    return resolve(variant, geometry('fixture_pca','coefficients',[ambient]),rank=rank,
        device='cpu',steps=8,preflight=True)[0]


def initialized(variant, *, image=False):
    torch.manual_seed(114)
    if image:
        config,_=resolve(variant,geometry('fixture','dataset',[4,4,1]),device='cpu',steps=8,preflight=True)
        model=build_model(variant,config,16).double()
    else:
        config=vector_config(variant)
        model=build_model(variant,config,7).double()
        with torch.no_grad():
            model.basis.copy_(torch.linalg.qr(torch.randn(7,2,dtype=torch.float64))[0])
            model.variances.copy_(torch.tensor([1.,3.]))
    with torch.no_grad(): model.core.output.weight.normal_(std=.001)
    model._lid_family=native_contracts()[variant]['family']
    model.eval()
    return model,config


@pytest.mark.parametrize('df',[5,128])
def test_radial_student_is_unit_rms_and_matches_pfgm_beta_prime(df):
    noise=student_unit_rms((100000,4),df,device='cpu',dtype=torch.float64,
        generator=torch.Generator().manual_seed(923)).numpy()
    np.testing.assert_allclose(np.mean(noise**2,axis=0),1.,atol=.05,rtol=0)
    statistic=kstest(np.sum(noise**2,axis=1)/(df-2),betaprime(2,df/2).cdf).statistic
    assert statistic<.009
    if df==5:
        # Shared mixing produces E[Z1^2 Z2^2]=3, versus1 for independent t's.
        assert np.mean(noise[:,0]**2*noise[:,1]**2)>1.7


@pytest.mark.parametrize('variant',['t_flowmatching','pfgmpp'])
def test_native_sampler_has_no_clipped_endpoint_masses(variant):
    config=vector_config(variant);family=native_contracts()[variant]['family']
    s=sample_rms_scale(40000,family,config,device='cpu',dtype=torch.float64,
        generator=torch.Generator().manual_seed(381)).numpy()
    k=math.sqrt(config.kernel_df/(config.kernel_df-2))
    assert np.all((s>config.sigma_min)&(s<config.sigma_max))
    if family=='student_t_flow':
        t=k/(k+s);low=k/(k+config.sigma_max);high=k/(k+config.sigma_min)
        uniform=(t-low)/(high-low)
    else:
        cdf=norm.cdf((np.log(s/k)-config.kernel_log_scale_mean)/config.kernel_log_scale_std)
        low=norm.cdf((math.log(config.sigma_min/k)-config.kernel_log_scale_mean)/config.kernel_log_scale_std)
        high=norm.cdf((math.log(config.sigma_max/k)-config.kernel_log_scale_mean)/config.kernel_log_scale_std)
        uniform=(cdf-low)/(high-low)
    assert kstest(uniform,'uniform').statistic<.012


def test_active_conditional_kernel_equals_full_ambient_posterior():
    torch.manual_seed(22)
    basis=torch.linalg.qr(torch.randn(7,2,dtype=torch.float64))[0]
    points=torch.randn(31,2,dtype=torch.float64)
    ambient=points@basis.T
    queries=torch.randn(9,7,dtype=torch.float64)
    scales=torch.linspace(.1,2.,9,dtype=torch.float64)
    for df in (5,128):
        conditional=conditional_rms_scale(queries,basis,scales,df)
        full_distance=(queries[:,None]-ambient[None]).square().sum(2)
        full_weights=torch.softmax(-.5*(df+7)*torch.log1p(full_distance/((df-2)*scales[:,None]**2)),1)
        active_distance=((queries@basis)[:,None]-points[None]).square().sum(2)
        df_active=df+5
        active_weights=torch.softmax(-.5*(df_active+2)*torch.log1p(active_distance/((df_active-2)*conditional[:,None]**2)),1)
        torch.testing.assert_close(full_weights@ambient,active_weights@points@basis.T,atol=3e-14,rtol=3e-14)
        wrong=torch.softmax(-.5*(df+2)*torch.log1p(active_distance/((df-2)*scales[:,None]**2)),1)
        assert float((wrong@points@basis.T-full_weights@ambient).abs().max())>.002


@pytest.mark.parametrize('variant',['t_flowmatching','pfgmpp'])
@pytest.mark.parametrize('image',[False,True])
def test_actual_field_native_trace_and_finite_difference_agree(variant,image):
    model,config=initialized(variant,image=image)
    n=16 if image else 7
    x=torch.randn(2,n,dtype=torch.float64);scale=.7
    response=exact_divergence(model,x,scale).detach()
    if not image:
        _,active=covariance_span_value_and_trace(model,x,scale)
        torch.testing.assert_close(active,response,rtol=2e-10,atol=2e-10)
    k=math.sqrt(config.kernel_df/(config.kernel_df-2));time=k/(k+scale)
    class Velocity(torch.nn.Module):
        def forward(self,y,t):return model.native_velocity(y,t)
    class Radial(torch.nn.Module):
        def forward(self,y,r):return model.poisson_radial_field(y,r)
    native_flow=n*time+time*(1-time)*exact_divergence(Velocity(),time*x,time)
    radius=scale*math.sqrt(config.kernel_df-2)
    native_poisson=n-radius*exact_divergence(Radial(),x,radius)
    torch.testing.assert_close(native_flow,response,rtol=2e-10,atol=2e-10)
    torch.testing.assert_close(native_poisson,response,rtol=2e-10,atol=2e-10)
    fd=torch.zeros(2,dtype=torch.float64)
    for j in range(n):
        delta=torch.zeros_like(x);delta[:,j]=1e-6
        fd+=((model(x+delta,scale)-model(x-delta,scale))[:,j]/2e-6).detach()
    torch.testing.assert_close(fd,response,rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize('variant',['t_flowmatching','pfgmpp'])
@pytest.mark.parametrize('readout',['full','fm_to_score','fixed_likelihood'])
def test_gaussian_lid_conversions_are_rejected(variant,readout):
    model,_=initialized(variant)
    with pytest.raises(ValueError,match='response only'):
        training.predict_lid(model,np.ones((2,7)),.5,family=native_contracts()[variant]['family'],
            readout=readout,divergence_backend='active_exact',trace_probes=0)
    assert readouts(variant)==('response',)


@pytest.mark.parametrize('variant',['t_flowmatching','pfgmpp'])
def test_loss_matches_native_noise_or_edm_objective_samplewise(variant):
    model,config=initialized(variant)
    family=native_contracts()[variant]['family']
    clean=torch.randn(8,2,dtype=torch.float64)@model.basis.T
    generator=torch.Generator().manual_seed(62)
    scale=sample_rms_scale(8,family,config,device='cpu',dtype=clean.dtype,generator=generator)
    noise=student_unit_rms(clean.shape,config.kernel_df,device='cpu',dtype=clean.dtype,generator=generator)
    k=math.sqrt(config.kernel_df/(config.kernel_df-2))
    sigma=scale/k
    canonical=clean+scale[:,None]*noise
    prediction=model(canonical,scale)
    if family=='student_t_flow':
        predicted_native_noise=(canonical-prediction)/sigma[:,None]
        expected=(predicted_native_noise-k*noise).square().mean()
    else:
        expected=(((sigma.square()+1)/sigma.square())[:,None]*(prediction-clean).square()).mean()
    actual=loss(family,model,clean,config,torch.Generator().manual_seed(62))
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('variant',['t_flowmatching','pfgmpp'])
def test_training_and_reload_use_common_budget_selection_and_exact_parameter_count(tmp_path,variant):
    torch.manual_seed(8)
    raw=torch.randn(64,2)@torch.linalg.qr(torch.randn(7,2))[0].T
    config=vector_config(variant)
    result=training.train_model(native_contracts()[variant]['family'],raw[:48],raw[48:],config,tmp_path/'model.pt')
    loaded=training.load_checkpoint(tmp_path/'model.pt',device='cpu')
    assert result.config.to_dict()==loaded.config.to_dict()
    for key,value in result.model.state_dict().items():
        torch.testing.assert_close(value,loaded.model.state_dict()[key],rtol=0,atol=0)
    assert result.metrics['steps_completed']==8
    expected=resolve(variant,geometry('fixture_pca','coefficients',[7]),rank=2,device='cpu',steps=8,preflight=True)[1]
    assert parameter_count(result.model)==expected['reference_parameters']
    a=training.predict_lid(result,raw[:5],.5,readout='response',divergence_backend='active_exact',trace_probes=0)
    b=training.predict_lid(loaded,raw[:5],.5,readout='response',divergence_backend='exact',trace_probes=0)
    np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
