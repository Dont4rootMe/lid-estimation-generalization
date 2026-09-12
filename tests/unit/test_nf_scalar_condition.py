"""NF condition information survives without any periodic feature expansion."""
from dataclasses import replace

import pytest
import torch

from experiments import fair_protocol as fair
from models import training
from models.normalizing_flow import ConditionalFlowConfig, ScaleConditionedRealNVP


@pytest.mark.parametrize('representation,shape',[
    ('coefficients',[3]), ('dataset',[1,4,4]), ('dataset',[4,4,3]),
])
def test_current_nf_reads_log_sigma_without_trigonometry(monkeypatch,representation,shape):
    geo=fair.geometry('fixture',representation,shape)
    cfg,receipt=fair.resolve('scale_conditioned_nf',geo,device='cpu')
    torch.manual_seed(41)
    model=fair.build_model('scale_conditioned_nf',cfg,geo['ambient_dim']).double()
    assert cfg.fourier_features==0
    assert not any('frequenc' in name for name,_ in model.named_buffers())
    assert fair.parameter_count(model)==receipt['actual_parameters']
    assert abs(receipt['nf_relative_gap'])<=.1
    # Nonzero deterministic weights make conditioning observable without fitting.
    if representation=='coefficients':
        outputs=[layer.conditioner[-1] for layer in model.couplings]
        first=[model.condition_embedding.projection[0]]
    else:
        outputs=[layer.conditioner.output for layer in model.couplings]
        first=[layer.conditioner.time[0] for layer in model.couplings]
    with torch.no_grad():
        for layer in outputs:layer.weight.normal_(0,.002)
    captured=[]
    handles=[layer.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone())) for layer in first]
    def forbidden(*args,**kwargs):raise AssertionError('NF executed a periodic feature operation')
    for name in ['sin','cos']:
        monkeypatch.setattr(torch,name,forbidden)
        monkeypatch.setattr(torch.Tensor,name,forbidden)
    x=torch.randn(2,geo['ambient_dim'],dtype=torch.float64)
    sigma=torch.tensor([.3,1.7],dtype=torch.float64,requires_grad=True)
    z,det=model.encode(x,sigma)
    restored,invdet=model.decode(z,sigma)
    torch.testing.assert_close(restored,x,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(det+invdet,torch.zeros_like(det),atol=1e-10,rtol=0)
    assert captured
    for value in captured:torch.testing.assert_close(value,sigma.detach().log()[:,None],atol=0,rtol=0)
    # Verify the learned condition path itself, separately from fixed input scaling.
    if representation=='coefficients':
        embedding=model.condition_embedding(sigma)
    else:
        embedding=model.couplings[0].conditioner.time(sigma.log()[:,None])
    derivative=torch.autograd.grad(embedding.sum(),sigma)[0]
    assert torch.isfinite(derivative).all() and (derivative.abs()>1e-8).all()
    for handle in handles:handle.remove()


def test_old_nf_architecture_reconstructs_explicit_legacy_frequencies():
    old=ConditionalFlowConfig(ambient_dim=3,hidden_dim=8,num_coupling_layers=2,
                              condition_dim=4,fourier_features=32,max_condition_frequency=1.)
    model=ScaleConditionedRealNVP(old).double()
    reloaded=ScaleConditionedRealNVP(ConditionalFlowConfig.from_mapping(old.to_dict())).double()
    reloaded.load_state_dict(model.state_dict())
    x=torch.tensor([[.2,.3,-.1]],dtype=torch.float64)
    torch.testing.assert_close(model.log_prob(x,.7),reloaded.log_prob(x,.7),atol=0,rtol=0)
    assert model.condition_embedding.frequencies.numel()==32
    with pytest.raises(RuntimeError):
        ScaleConditionedRealNVP(replace(old,fourier_features=0)).load_state_dict(model.state_dict())


def test_fresh_native_nf_rejects_old_frequency_configuration_before_training(tmp_path):
    geo=fair.geometry('fixture','coefficients',[3])
    cfg,_=fair.resolve('scale_conditioned_nf',geo,device='cpu')
    with pytest.raises(ValueError,match='scalar log-sigma'):
        training.train_model('scale_conditioned_normalizing_flow',None,None,
                             replace(cfg,fourier_features=32),tmp_path/'forbidden.pt')
    assert not list(tmp_path.iterdir())
