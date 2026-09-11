"""Full-coordinate training and derivatives, including low-rank clean data."""
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from experiments.fair_protocol import geometry,resolve,build_model,native_contracts
from experiments.lambda_repair_eval import CanonicalPosterior
from models import training
from models.neural_fields import exact_divergence

torch.set_num_threads(2)


@pytest.mark.parametrize('variant',list(native_contracts()))
def test_full_vector_trains_on_every_coordinate_and_reloads(tmp_path,variant,monkeypatch):
    import models.preconditioned_field as legacy
    def prohibited(*args,**kwargs): raise AssertionError('fitted support is prohibited')
    monkeypatch.setattr(legacy,'training_covariance_span',prohibited)
    cfg,meta=resolve(variant,geometry('fixture','coefficients',[6]),
        device='cpu',steps=4,preflight=True)
    # Clean observations occupy one line. The neural model must still learn
    # every one of the six corruption/output coordinates, without a projection.
    x=np.zeros((40,6),dtype=np.float32);x[:,0]=np.linspace(-2,2,40)
    result=training.train_model(native_contracts()[variant]['family'],x[:32],x[32:],cfg,tmp_path/'model.pt')
    loaded=training.load_checkpoint(tmp_path/'model.pt',device='cpu')
    assert result.metrics['steps_completed']==4
    assert result.config.field_projection_rank is None
    assert not any('basis' in k or 'variances' in k for k in result.model.state_dict())
    assert sum(p.numel() for p in result.model.parameters())==meta['actual_parameters']
    for key,value in result.model.state_dict().items():
        torch.testing.assert_close(value,loaded.model.state_dict()[key],rtol=0,atol=0)
    if variant!='scale_conditioned_nf':
        model=result.model.double();family=training._canonical_family(native_contracts()[variant]['family'])
        q=torch.randn(2,6,dtype=torch.float64,requires_grad=True)
        adapter=CanonicalPosterior(SimpleNamespace(model=model,config=cfg,family=family),.3)
        value=adapter(q)
        assert value.shape==q.shape
        # No output coordinate is analytically forced into the clean data span.
        assert torch.all(value[:,1:].abs().sum(0)>1e-8)
        derivative=torch.autograd.grad(value.square().sum(),q)[0]
        assert torch.all(derivative.abs().sum(0)>1e-8)


@pytest.mark.parametrize('variant',['t_flowmatching','pfgmpp'])
def test_ambient_native_trace_matches_velocity_radial_and_finite_difference(variant):
    cfg,_=resolve(variant,geometry('fixture','coefficients',[6]),device='cpu',steps=4,preflight=True)
    model=build_model(variant,cfg,6).double()
    with torch.no_grad():model.core.output.weight.normal_(std=.01)
    x=torch.randn(2,6,dtype=torch.float64);scale=.7
    response=exact_divergence(model,x,scale).detach()
    k=math.sqrt(cfg.kernel_df/(cfg.kernel_df-2));time=k/(k+scale)
    class Velocity(torch.nn.Module):
        def forward(self,y,t):return model.native_velocity(y,t)
    class Radial(torch.nn.Module):
        def forward(self,y,r):return model.poisson_radial_field(y,r)
    radius=scale*math.sqrt(cfg.kernel_df-2)
    torch.testing.assert_close(6*time+time*(1-time)*exact_divergence(Velocity(),time*x,time),response,rtol=2e-10,atol=2e-10)
    torch.testing.assert_close(6-radius*exact_divergence(Radial(),x,radius),response,rtol=2e-10,atol=2e-10)
    fd=torch.zeros(2,dtype=torch.float64)
    for j in range(6):
        delta=torch.zeros_like(x);delta[:,j]=1e-6
        fd+=((model(x+delta,scale)-model(x-delta,scale))[:,j]/2e-6).detach()
    torch.testing.assert_close(fd,response,rtol=2e-6,atol=2e-6)


def test_full_vector_nf_inverse_and_full_jacobian():
    cfg,_=resolve('scale_conditioned_nf',geometry('fixture','coefficients',[6]),device='cpu',steps=4,preflight=True)
    model=build_model('scale_conditioned_nf',cfg,6).double()
    with torch.no_grad():
        for coupling in model.couplings:
            coupling.conditioner[-1].weight.normal_(std=.003)
            coupling.conditioner[-1].bias.normal_(std=.003)
    z=torch.randn(2,6,dtype=torch.float64);scales=torch.tensor([.2,2.3],dtype=z.dtype)
    x,forward=model.decode(z,scales);back,inverse=model.encode(x,scales)
    torch.testing.assert_close(back,z,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(forward,-inverse,rtol=1e-10,atol=1e-10)
    jac=torch.autograd.functional.jacobian(lambda y:model.encode(y[None],.2)[0][0],x[0])
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],inverse[0],rtol=1e-9,atol=1e-9)
