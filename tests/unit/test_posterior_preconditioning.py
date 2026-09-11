from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.training import TrainingConfig, train_model, load_checkpoint, predict_lid
from models.vp_baseline import ConditionedBottleneckMLP, PreconditionedLogNoiseMLP, PreconditionedLogNoiseNoInputSkipMLP


def test_native_loss_and_parameter_gradient_equivalence():
    torch.manual_seed(932)
    model = PreconditionedLogNoiseMLP(12, (8, 4), 8).double()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.05)
    clean, noise = torch.randn(9, 12, dtype=torch.float64), torch.randn(9, 12, dtype=torch.float64)
    scales = torch.logspace(-8, 6, 9, base=2, dtype=torch.float64)
    stable = model.residual_loss(clean, scales, noise)
    direct = (((model(clean + scales[:, None] * noise, scales.log()) - clean) / scales[:, None]) ** 2).mean()
    torch.testing.assert_close(stable, direct, atol=1e-12, rtol=1e-12)
    left = torch.autograd.grad(stable, tuple(model.parameters()))
    right = torch.autograd.grad(direct, tuple(model.parameters()))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)


def test_gaussian_initialization_same_capacity_and_full_jacobian():
    torch.manual_seed(4)
    legacy = ConditionedBottleneckMLP(12, (8, 4), 8).double()
    torch.manual_seed(4)
    fixed = PreconditionedLogNoiseMLP(12, (8, 4), 8).double()
    assert sum(p.numel() for p in legacy.parameters()) == sum(p.numel() for p in fixed.parameters())
    for name, value in legacy.state_dict().items():
        if not name.startswith('decoder.1.'):
            torch.testing.assert_close(value, fixed.state_dict()[name], atol=0, rtol=0)
    point = torch.randn(12, dtype=torch.float64)
    for scale in (1/256, 1, 64):
        log_scale = torch.tensor(np.log(scale), dtype=torch.float64)
        q = fixed(point[None], log_scale)[0]
        jac = torch.autograd.functional.jacobian(lambda x: fixed(x[None], log_scale)[0], point)
        torch.testing.assert_close(q, point/(1+scale**2))
        torch.testing.assert_close(jac, torch.eye(12, dtype=torch.float64)/(1+scale**2))


def tiny_config():
    return TrainingConfig(
        seed=0, device='cpu', training_mode='fixed_steps_v1', steps=4,
        warmup_steps=0, validation_interval_steps=2, batch_size=8,
        learning_rate=2e-4, weight_decay=1e-6, gradient_clip_norm=1,
        early_stopping_patience=None, field_hidden_sizes=(8, 4), time_embedding_dim=8,
        flow_variant_id='posterior_log_noise_affine_flow', flow_schedule='log_noise',
        flow_parameterization='posterior_mean', flow_conditioning='log_noise_ratio',
        flow_scale_sampling='log_uniform_noise_ratio', flow_loss_weighting='posterior_bias_equivalent',
        flow_noise_ratio_min=1/256, flow_noise_ratio_max=64,
    )


@pytest.mark.parametrize('mode', [None, 'unit_rms_gaussian_v1', 'unit_rms_gaussian_no_input_skip_v1'])
def test_training_reload_readout_roundtrip(tmp_path, mode):
    config = replace(tiny_config(), posterior_preconditioning=mode)
    rng = np.random.default_rng(2)
    train, hold = rng.normal(size=(32,12)), rng.normal(size=(8,12))
    result = train_model('independent_affine_flow', train, hold, config, tmp_path/'model.pt')
    loaded = load_checkpoint(tmp_path/'model.pt')
    assert isinstance(loaded.model, PreconditionedLogNoiseMLP) == (mode is not None)
    for readout in ('full', 'response', 'fm_to_score'):
        a = predict_lid(result, hold[:2], 16, readout=readout, divergence_backend='exact')
        b = predict_lid(loaded, hold[:2], 16, readout=readout, divergence_backend='exact')
        np.testing.assert_array_equal(a,b)


def test_incompatible_condition_rejected():
    with pytest.raises(ValueError, match='requires normalized'):
        replace(tiny_config(), posterior_preconditioning='unit_rms_gaussian_v1', flow_schedule='rectified')


def test_no_input_skip_equals_zero_direct_matrix_and_freezes_its_gradient():
    torch.manual_seed(3)
    model = PreconditionedLogNoiseNoInputSkipMLP(12, (8,4), 8).double()
    torch.nn.init.normal_(model.decoder[-1].weight, std=.1)
    reference = PreconditionedLogNoiseMLP(12, (8,4), 8).double()
    reference.load_state_dict(model.state_dict())
    with torch.no_grad():
        reference.decoder[-1].weight[:,:12].zero_()
    query = torch.randn(3,12,dtype=torch.float64)
    condition = torch.tensor([-3.,0.,4.],dtype=torch.float64)
    torch.testing.assert_close(model(query,condition), reference(query,condition))
    model(query,condition).square().mean().backward()
    assert torch.count_nonzero(model.decoder[-1].weight.grad[:,:12]) == 0
    assert torch.count_nonzero(model.encoder[0].weight.grad) > 0
