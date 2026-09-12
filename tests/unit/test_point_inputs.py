"""Guard the author-required point-only spatial inputs, without optimizer steps."""
from dataclasses import replace

import pytest
import torch

from experiments.fair_protocol import build_model, geometry, native_contracts, resolve
from models import training
from models.native_tasks import sample_batch
from models.point_residual_field import PointResidualCore
from models.spectral_residual_field import SpectralResidualCore


FIELDS=tuple(v for v in native_contracts() if v!='scale_conditioned_nf')


def config(variant):
    cfg,_=resolve(variant,geometry('points','coefficients',[3]),device='cpu',steps=4,preflight=True)
    return replace(cfg,field_residual_width=16)


@pytest.mark.parametrize('variant',FIELDS)
def test_every_native_vector_core_consumes_the_point_itself(variant):
    cfg=config(variant)
    model=build_model(variant,cfg,3)
    x=torch.tensor([[.3,-.7,1.4],[-1.2,.5,.8]],requires_grad=True)
    cores=[model.core]+([model.backward_core] if variant=='schrodinger_bridge' else [])
    for core in cores:
        assert type(core) is PointResidualCore and core.input.in_features==3
        captured=[]
        handle=core.input.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        model.raw(x,.4,core)
        handle.remove()
        assert len(captured)==1 and captured[0] is x
        assert not any('spatial_frequenc' in name for name,_ in core.named_buffers())
        # Conditioning is separate from the spatial input and remains present.
        assert 'condition_frequencies' in dict(core.named_buffers())


@pytest.mark.parametrize('variant',[v for v in FIELDS if v!='schrodinger_bridge'])
def test_feature_change_preserves_native_corruptions_targets_and_weights(variant):
    cfg=config(variant)
    old=replace(cfg,field_backbone='spectral_residual_v1')
    x=torch.tensor([[1.,2.,-1.],[.5,-2.,.3]])
    new_batch=sample_batch(x,cfg,torch.Generator().manual_seed(416))
    old_batch=sample_batch(x,old,torch.Generator().manual_seed(416))
    for actual,expected in zip(new_batch,old_batch):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_old_spectral_weights_are_explicit_and_never_silently_converted():
    cfg=config('t_flowmatching')
    current=build_model('t_flowmatching',cfg,3)
    old_cfg=replace(cfg,field_backbone='spectral_residual_v1')
    old=build_model('t_flowmatching',old_cfg,3)
    restored=build_model('t_flowmatching',training.TrainingConfig.from_mapping(old_cfg.to_dict()),3)
    assert type(old.core) is SpectralResidualCore
    restored.load_state_dict(old.state_dict(),strict=True)
    x=torch.randn(2,3)
    torch.testing.assert_close(old(x,.3),restored(x,.3),rtol=0,atol=0)
    with pytest.raises(RuntimeError,match='size mismatch'):
        current.load_state_dict(old.state_dict(),strict=True)


def test_new_training_rejects_a_historical_spatial_fourier_config(tmp_path):
    cfg=replace(config('t_flowmatching'),field_backbone='spectral_residual_v1')
    destination=tmp_path/'forbidden.pt'
    with pytest.raises(ValueError,match='raw point inputs'):
        training.train_model('student_t_flow',None,None,cfg,destination)
    assert not destination.exists()
