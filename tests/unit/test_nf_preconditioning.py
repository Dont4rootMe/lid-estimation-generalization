from dataclasses import replace
import numpy as np
import torch
import pytest
from models.normalizing_flow import ConditionalFlowConfig, fixed_point_lid
from models.preconditioned_nf import CovariancePreconditionedNF
from models.training import TrainingConfig, train_model, load_checkpoint


def test_covariance_nf_inverse_logdet_density_and_scale_derivative():
    cfg=TrainingConfig(num_coupling_layers=2,conditioner_depth=1,log_scale_limit=2.,
        epsilon_min=.001,epsilon_max=100.,field_preconditioning='covariance_span_v1',field_projection_rank=3)
    model=CovariancePreconditionedNF(ConditionalFlowConfig(6,hidden_dim=12,num_coupling_layers=2),cfg).double()
    with torch.no_grad():
        model.basis[:3]=torch.eye(3,dtype=torch.float64)
        model.normal_basis[3:]=torch.eye(3,dtype=torch.float64)
        model.variances.copy_(torch.tensor([.2,2.,20.],dtype=torch.float64))
    x=torch.randn(4,6,dtype=torch.float64)
    for s in [.00390625,.1,2.,64.]:
        z,ld=model.encode(x,s);restored,ild=model.decode(z,s)
        torch.testing.assert_close(restored,x,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(ld,-ild,rtol=1e-12,atol=1e-12)
        variance=torch.cat((model.variances+s*s,torch.full((3,),s*s,dtype=torch.float64)))
        expected=-.5*(x.square()/variance+variance.log()+np.log(2*np.pi)).sum(1)
        torch.testing.assert_close(model.log_prob(x,s),expected,rtol=1e-12,atol=1e-12)
        expected_lid=6-(s*s/variance).sum()+((s*x/variance).square()).sum(1)
        torch.testing.assert_close(fixed_point_lid(model,x,s),expected_lid,rtol=1e-11,atol=1e-11)
    jac=torch.func.jacrev(lambda q:model.encode(q[None],.3)[0][0])(x[0])
    torch.testing.assert_close(torch.linalg.slogdet(jac).logabsdet,model.encode(x[:1],.3)[1][0])


@pytest.mark.parametrize('rank',[3,6])
def test_covariance_nf_train_and_checkpoint(tmp_path,rank):
    rng=np.random.default_rng(17);x=np.zeros((96,6),dtype=np.float32);x[:,:rank]=rng.normal(size=(96,rank))
    cfg=TrainingConfig(hidden_dim=12,num_coupling_layers=2,conditioner_depth=1,log_scale_limit=2.,
        epsilon_min=.01,epsilon_max=64.,field_preconditioning='covariance_span_v1',device='cpu',
        training_mode='fixed_steps_v1',steps=4,warmup_steps=0,validation_interval_steps=2,
        batch_size=16,early_stopping_patience=None)
    trained=train_model('scale_conditioned_nf',x[:80],x[80:],cfg,tmp_path/'model.pt')
    restored=load_checkpoint(tmp_path/'model.pt',device='cpu')
    assert restored.config.field_projection_rank==rank
    for name,value in trained.model.state_dict().items():
        torch.testing.assert_close(value,restored.model.state_dict()[name],rtol=0,atol=0)
