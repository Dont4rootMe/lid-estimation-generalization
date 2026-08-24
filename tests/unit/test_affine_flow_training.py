from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

import models.training as training_module
from models.affine_flow import (
    AffineFlowSpec,
    affine_flow_contract,
    affine_fm_to_score_full,
    affine_full_from_posterior,
    affine_interpolant_and_target,
    affine_response_from_posterior_divergence,
    affine_response_from_velocity_divergence,
    affine_schedule_state,
    flow_matching_loss_weights,
    posterior_divergence_to_marginal_score_divergence,
    posterior_divergence_to_velocity_divergence,
    posterior_to_marginal_score,
    posterior_to_velocity,
    sample_noise_ratio,
    velocity_divergence_to_posterior_divergence,
    velocity_to_posterior,
)
from models.training import (
    TrainingConfig,
    independent_affine_flow_matching_loss,
    load_checkpoint,
    predict_affine_primitives,
    predict_lid,
    train_model,
)

VARIANTS = (
    ("direct_rectified_flow", "rectified_linear", "direct_velocity"),
    ("posterior_rectified_flow", "rectified_linear", "posterior_mean"),
    ("direct_log_noise_affine_flow", "log_noise", "direct_velocity"),
    ("posterior_log_noise_affine_flow", "log_noise", "posterior_mean"),
    ("direct_vp_trigonometric_flow", "vp_trigonometric", "direct_velocity"),
    ("posterior_vp_trigonometric_flow", "vp_trigonometric", "posterior_mean"),
)


def _spec(variant_id: str, schedule: str, parameterization: str) -> AffineFlowSpec:
    return AffineFlowSpec.from_mapping(
        {
            "variant_id": variant_id,
            "schedule": schedule,
            "parameterization": parameterization,
            "conditioning": "log_noise_ratio",
            "scale_sampling": "log_uniform_noise_ratio",
            "loss_weighting": "posterior_bias_equivalent",
            "noise_ratio_min": 0.01,
            "noise_ratio_max": 1.0,
        }
    )


def _config(
    variant_id: str, schedule: str, parameterization: str, *, seed: int = 7
) -> TrainingConfig:
    return TrainingConfig(
        seed=seed,
        device="cpu",
        epochs=1,
        batch_size=8,
        learning_rate=2.0e-3,
        weight_decay=0.0,
        hidden_dim=16,
        depth=1,
        time_embedding_dim=8,
        validation_interval=1,
        early_stopping_patience=None,
        gradient_clip_norm=1.0,
        fourier_features=4,
        max_condition_frequency=10.0,
        normalize=True,
        flow_variant_id=variant_id,
        flow_schedule=schedule,
        flow_parameterization=parameterization,
        flow_conditioning="log_noise_ratio",
        flow_scale_sampling="log_uniform_noise_ratio",
        flow_loss_weighting="posterior_bias_equivalent",
        flow_noise_ratio_min=0.01,
        flow_noise_ratio_max=1.0,
    )


@pytest.mark.parametrize("schedule", [item[1] for item in VARIANTS[::2]])
def test_schedule_state_has_exact_noise_ratio_and_endpoint_orientation(
    schedule: str,
) -> None:
    noise_ratio = torch.tensor([0.02, 0.3, 1.0], dtype=torch.float64)
    state = affine_schedule_state(noise_ratio, schedule)
    torch.testing.assert_close(state.beta / state.alpha, noise_ratio)
    torch.testing.assert_close(
        state.beta_derivative / state.beta - state.alpha_derivative / state.alpha,
        state.log_noise_ratio_derivative,
    )
    assert torch.isfinite(state.native_time).all()
    assert torch.isfinite(state.log_noise_ratio_derivative).all()
    if schedule == "log_noise":
        assert torch.all(state.log_noise_ratio_derivative > 0)
        torch.testing.assert_close(state.native_time, torch.log(noise_ratio))
        torch.testing.assert_close(
            state.alpha_derivative, torch.zeros_like(state.alpha)
        )
        torch.testing.assert_close(state.beta_derivative, state.beta)
    elif schedule == "rectified_linear":
        assert torch.all(state.log_noise_ratio_derivative < 0)
        torch.testing.assert_close(state.native_time, state.alpha)
        torch.testing.assert_close(state.alpha_derivative, torch.ones_like(state.alpha))
        torch.testing.assert_close(state.beta_derivative, -torch.ones_like(state.beta))
    else:
        assert torch.all(state.log_noise_ratio_derivative < 0)
        angular_rate = math.pi / 2.0
        torch.testing.assert_close(state.alpha_derivative, angular_rate * state.beta)
        torch.testing.assert_close(state.beta_derivative, -angular_rate * state.alpha)


@pytest.mark.parametrize("variant_id,schedule,parameterization", VARIANTS)
def test_variant_contract_seals_schedule_parameterization_and_orientation(
    variant_id: str, schedule: str, parameterization: str
) -> None:
    contract = affine_flow_contract(_spec(variant_id, schedule, parameterization))
    assert contract["variant_id"] == variant_id
    assert contract["schedule"] == schedule
    assert contract["parameterization"] == parameterization
    assert contract["scale_semantics"] == "noise_ratio_lambda=beta/alpha"
    assert contract["data_endpoint"].endswith("lambda->0")
    if schedule == "log_noise":
        assert contract["native_orientation"] == "data_to_noise_as_u_increases"
    else:
        assert contract["native_orientation"] == "source_to_data_as_t_increases"


@pytest.mark.parametrize("variant_id,schedule,parameterization", VARIANTS)
def test_affine_velocity_posterior_and_divergence_round_trip(
    variant_id: str, schedule: str, parameterization: str
) -> None:
    del variant_id, parameterization
    generator = torch.Generator().manual_seed(91)
    point = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    posterior = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    posterior_divergence = torch.randn(5, generator=generator, dtype=torch.float64)
    state = affine_schedule_state(torch.full((5,), 0.2, dtype=torch.float64), schedule)
    velocity = posterior_to_velocity(posterior, point, state)
    velocity_divergence = posterior_divergence_to_velocity_divergence(
        posterior_divergence, state, ambient_dim=4
    )
    recovered_posterior = velocity_to_posterior(velocity, point, state)
    recovered_divergence = velocity_divergence_to_posterior_divergence(
        velocity_divergence, state, ambient_dim=4
    )
    torch.testing.assert_close(recovered_posterior, posterior)
    torch.testing.assert_close(recovered_divergence, posterior_divergence)
    torch.testing.assert_close(
        affine_response_from_velocity_divergence(
            velocity_divergence, state, ambient_dim=4
        ),
        affine_response_from_posterior_divergence(posterior_divergence, state),
    )


@pytest.mark.parametrize("variant_id,schedule,parameterization", VARIANTS)
def test_full_and_fm_to_score_are_identical_for_gaussian_affine_channel(
    variant_id: str, schedule: str, parameterization: str
) -> None:
    del variant_id, parameterization
    generator = torch.Generator().manual_seed(92)
    channel_point = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    posterior = 0.7 * channel_point
    posterior_divergence = torch.full((6,), 2.1, dtype=torch.float64)
    state = affine_schedule_state(torch.full((6,), 0.15, dtype=torch.float64), schedule)
    evaluation_point = state.alpha[:, None] * channel_point
    score = posterior_to_marginal_score(posterior, evaluation_point, state)
    score_divergence = posterior_divergence_to_marginal_score_divergence(
        posterior_divergence, state, ambient_dim=3
    )
    full = affine_full_from_posterior(
        posterior, posterior_divergence, channel_point, state
    )
    converted = affine_fm_to_score_full(score, score_divergence, state, ambient_dim=3)
    torch.testing.assert_close(converted, full, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("variant_id,schedule,parameterization", VARIANTS)
def test_posterior_bias_equivalent_loss_weight_is_exact(
    variant_id: str, schedule: str, parameterization: str
) -> None:
    state = affine_schedule_state(
        torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64), schedule
    )
    spec = _spec(variant_id, schedule, parameterization)
    actual = flow_matching_loss_weights(state, spec)
    if parameterization == "posterior_mean":
        expected = state.noise_ratio.pow(-2)
    else:
        expected = (
            state.alpha * state.log_noise_ratio_derivative * state.noise_ratio
        ).pow(-2)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.all(actual > 0)
    if variant_id == "direct_vp_trigonometric_flow":
        torch.testing.assert_close(actual, state.alpha.square() / (math.pi / 2.0) ** 2)


@pytest.mark.parametrize("variant_id,schedule,parameterization", VARIANTS)
def test_training_objective_applies_declared_per_example_weighting(
    variant_id: str, schedule: str, parameterization: str
) -> None:
    class ZeroField(nn.Module):
        def forward(self, inputs, condition):  # type: ignore[no-untyped-def]
            del condition
            return torch.zeros_like(inputs)

    spec = _spec(variant_id, schedule, parameterization)
    data = torch.tensor([[0.2, -0.4], [0.8, 0.1], [-0.3, 0.5]], dtype=torch.float64)
    actual_generator = torch.Generator().manual_seed(1234)
    actual = independent_affine_flow_matching_loss(
        ZeroField(), data, spec=spec, generator=actual_generator
    )

    reference_generator = torch.Generator().manual_seed(1234)
    noise_ratio = sample_noise_ratio(
        data.shape[0], spec=spec, data=data, generator=reference_generator
    )
    state = affine_schedule_state(noise_ratio, schedule)
    noise = torch.randn(
        data.shape,
        device=data.device,
        dtype=data.dtype,
        generator=reference_generator,
    )
    _, target = affine_interpolant_and_target(
        data, noise, state, parameterization=parameterization
    )
    expected = torch.mean(
        flow_matching_loss_weights(state, spec) * target.square().flatten(1).mean(dim=1)
    )
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual)


def test_affine_contract_rejects_partial_or_mismatched_hydra_identity() -> None:
    with pytest.raises(ValueError, match="complete block"):
        replace(TrainingConfig(), flow_variant_id="posterior_rectified_flow")
    with pytest.raises(ValueError, match="requires schedule/parameterization"):
        _spec(
            "posterior_rectified_flow",
            "log_noise",
            "posterior_mean",
        )
    value = _spec(
        "posterior_rectified_flow", "rectified_linear", "posterior_mean"
    ).to_dict()
    value["conditioning"] = "native_time"
    with pytest.raises(ValueError, match="require log_noise_ratio conditioning"):
        AffineFlowSpec.from_mapping(value)
    value = _spec(
        "posterior_rectified_flow", "rectified_linear", "posterior_mean"
    ).to_dict()
    value["loss_weighting"] = "uniform"
    with pytest.raises(ValueError, match="posterior_bias_equivalent"):
        AffineFlowSpec.from_mapping(value)
    with pytest.raises(ValueError, match="forbids inactive family settings"):
        replace(
            _config(
                "posterior_rectified_flow",
                "rectified_linear",
                "posterior_mean",
            ),
            time_min=0.1,
            time_max=0.9,
        )


def test_non_affine_family_rejects_complete_but_inactive_flow_block(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(46)
    data = rng.normal(size=(8, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="cannot carry inactive"):
        train_model(
            "diffusion",
            data,
            data,
            _config(
                "posterior_rectified_flow",
                "rectified_linear",
                "posterior_mean",
            ),
            tmp_path / "inactive.ckpt",
        )


class _AnalyticGaussianAffineField(nn.Module):
    def __init__(self, spec: AffineFlowSpec, *, ambient_dim: int = 3) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.config = SimpleNamespace(ambient_dim=ambient_dim)
        self._lid_family = "independent_affine_flow"
        self._lid_affine_spec = spec

    def forward(self, inputs, condition):  # type: ignore[no-untyped-def]
        noise_ratio = torch.exp(torch.as_tensor(condition, dtype=inputs.dtype))
        state = affine_schedule_state(noise_ratio, self._lid_affine_spec.schedule)
        # X~N(0,I): E[X|Y=y] = alpha*y/(alpha^2+beta^2).
        coefficient = state.alpha / (state.alpha.square() + state.beta.square())
        posterior = coefficient[:, None] * inputs + self.anchor * 0.0
        if self._lid_affine_spec.parameterization == "posterior_mean":
            return posterior
        return posterior_to_velocity(posterior, inputs, state)


@pytest.mark.parametrize("variant_id,schedule,parameterization", VARIANTS)
def test_predict_affine_primitives_exact_hutchinson_and_readout_equivalence(
    variant_id: str, schedule: str, parameterization: str
) -> None:
    spec = _spec(variant_id, schedule, parameterization)
    model = _AnalyticGaussianAffineField(spec)
    query = np.array([[0.1, -0.2, 0.3], [0.4, 0.0, -0.1]], dtype=np.float64)
    exact = predict_affine_primitives(
        model,
        query,
        0.2,
        divergence_backend="exact",
        batch_size=1,
    )
    hutch = predict_affine_primitives(
        model,
        query,
        0.2,
        divergence_backend="hutchinson",
        trace_probes=2,
        trace_seed=19,
        batch_size=1,
    )
    np.testing.assert_allclose(
        hutch.posterior_divergence,
        exact.posterior_divergence,
        rtol=1e-12,
        atol=1e-12,
    )
    full = predict_lid(
        model,
        query,
        0.2,
        readout="full",
        divergence_backend="exact",
    )
    converted = predict_lid(
        model,
        query,
        0.2,
        readout="fm_to_score",
        divergence_backend="exact",
    )
    np.testing.assert_allclose(converted, full, rtol=1e-11, atol=1e-11)
    assert exact.variant_id == variant_id
    assert exact.noise_ratio == pytest.approx(0.2)
    assert np.isfinite(exact.velocity_divergence).all()
    np.testing.assert_allclose(
        exact.velocity_divergence,
        exact.velocity_divergence_from_posterior,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        hutch.velocity_divergence,
        hutch.velocity_divergence_from_posterior,
        rtol=1e-12,
        atol=1e-12,
    )
    assert exact.divergence_backend == "exact"
    assert exact.trace_probe_kind == "exact"
    assert exact.trace_seed is None
    assert exact.trace_probes == 0
    assert not exact.shared_posterior_velocity_probes
    assert exact.primary_trace_field == "posterior_mean"
    assert hutch.divergence_backend == "hutchinson"
    assert hutch.trace_probe_kind == "rademacher"
    assert hutch.trace_seed == 19
    assert hutch.trace_probes == 2
    assert hutch.shared_posterior_velocity_probes == (
        parameterization == "direct_velocity"
    )
    if parameterization == "direct_velocity":
        np.testing.assert_allclose(
            exact.model_output_divergence, exact.velocity_divergence
        )
        assert exact.velocity_divergence_source == "raw_model_trace"
    else:
        np.testing.assert_allclose(
            exact.model_output_divergence, exact.posterior_divergence
        )
        assert exact.velocity_divergence_source == "derived_from_posterior_trace"


@pytest.mark.parametrize("backend", ["exact", "hutchinson"])
def test_direct_variant_traces_raw_velocity_independently_with_shared_probes(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec("direct_log_noise_affine_flow", "log_noise", "direct_velocity")
    model = _AnalyticGaussianAffineField(spec, ambient_dim=2)
    calls: list[tuple[nn.Module, object | None]] = []

    def fake_exact(field, inputs, condition, **kwargs):  # type: ignore[no-untyped-def]
        del condition, kwargs
        calls.append((field, None))
        value = 11.0 if field is model else 7.0
        return torch.full(
            (inputs.shape[0],), value, device=inputs.device, dtype=inputs.dtype
        )

    def fake_hutch(field, inputs, condition, **kwargs):  # type: ignore[no-untyped-def]
        del condition
        probes = kwargs["probes"]
        assert probes is not None
        assert kwargs.get("generator") is None
        assert kwargs.get("seed") is None
        calls.append((field, probes))
        value = 11.0 if field is model else 7.0
        return torch.full(
            (inputs.shape[0],), value, device=inputs.device, dtype=inputs.dtype
        )

    if backend == "exact":
        monkeypatch.setattr(training_module, "exact_divergence", fake_exact)
    else:
        monkeypatch.setattr(training_module, "hutchinson_divergence", fake_hutch)
    prediction = predict_affine_primitives(
        model,
        np.asarray([[0.2, -0.3]], dtype=np.float32),
        0.2,
        divergence_backend=backend,  # type: ignore[arg-type]
        trace_probes=3,
        trace_seed=23,
    )
    assert len(calls) == 2
    assert calls[0][0] is not model
    assert calls[1][0] is model
    if backend == "hutchinson":
        assert calls[0][1] is calls[1][1]
    np.testing.assert_allclose(prediction.model_output_divergence, [11.0])
    np.testing.assert_allclose(prediction.velocity_divergence, [11.0])
    assert not np.allclose(
        prediction.velocity_divergence,
        prediction.velocity_divergence_from_posterior,
    )


def test_posterior_variant_uses_one_raw_posterior_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec("posterior_rectified_flow", "rectified_linear", "posterior_mean")
    model = _AnalyticGaussianAffineField(spec, ambient_dim=2)
    calls: list[nn.Module] = []

    def fake_exact(field, inputs, condition, **kwargs):  # type: ignore[no-untyped-def]
        del condition, kwargs
        calls.append(field)
        return torch.full(
            (inputs.shape[0],), 5.0, device=inputs.device, dtype=inputs.dtype
        )

    monkeypatch.setattr(training_module, "exact_divergence", fake_exact)
    prediction = predict_affine_primitives(
        model,
        np.asarray([[0.2, -0.3]], dtype=np.float32),
        0.2,
        divergence_backend="exact",
    )
    assert calls == [model]
    np.testing.assert_allclose(prediction.model_output_divergence, [5.0])
    np.testing.assert_allclose(prediction.posterior_divergence, [5.0])
    assert prediction.velocity_divergence_source == "derived_from_posterior_trace"


@pytest.mark.parametrize("variant_id,schedule,parameterization", VARIANTS)
def test_all_affine_variants_train_checkpoint_load_and_predict(
    variant_id: str,
    schedule: str,
    parameterization: str,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(44)
    train = rng.normal(size=(24, 3)).astype(np.float32)
    validation = rng.normal(size=(12, 3)).astype(np.float32)
    checkpoint = tmp_path / variant_id / "model.ckpt"
    result = train_model(
        "independent_affine_flow",
        train,
        validation,
        _config(variant_id, schedule, parameterization),
        checkpoint,
    )
    assert result.family == "independent_affine_flow"
    assert result.model_contract["variant_id"] == variant_id
    assert result.model_contract["network_output"] in {
        "E[X|Y]",
        "native_schedule_velocity",
    }
    loaded = load_checkpoint(checkpoint)
    assert loaded.model_contract == result.model_contract
    response = predict_lid(
        loaded,
        validation[:3],
        0.2,
        readout="response",
        divergence_backend="exact",
    )
    full = predict_lid(
        loaded,
        validation[:3],
        0.2,
        readout="full",
        divergence_backend="exact",
    )
    converted = predict_lid(
        loaded,
        validation[:3],
        0.2,
        readout="fm_to_score",
        divergence_backend="exact",
    )
    assert response.shape == full.shape == converted.shape == (3,)
    assert np.isfinite(response).all()
    assert np.isfinite(full).all()
    np.testing.assert_allclose(converted, full, rtol=1e-5, atol=1e-5)


def test_affine_checkpoint_rejects_variant_contract_tampering(tmp_path: Path) -> None:
    rng = np.random.default_rng(45)
    train = rng.normal(size=(16, 2)).astype(np.float32)
    checkpoint = tmp_path / "affine.ckpt"
    train_model(
        "independent_affine_flow",
        train,
        train[:8],
        _config(
            "posterior_rectified_flow",
            "rectified_linear",
            "posterior_mean",
        ),
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_contract"]["variant_id"] = "direct_rectified_flow"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="model_contract mismatch"):
        load_checkpoint(checkpoint)


def test_affine_checkpoint_rejects_architecture_tampering(tmp_path: Path) -> None:
    rng = np.random.default_rng(47)
    train = rng.normal(size=(16, 2)).astype(np.float32)
    checkpoint = tmp_path / "affine-architecture.ckpt"
    train_model(
        "independent_affine_flow",
        train,
        train[:8],
        _config(
            "direct_log_noise_affine_flow",
            "log_noise",
            "direct_velocity",
        ),
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["architecture"]["condition_transform"] = "log"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="architecture does not match"):
        load_checkpoint(checkpoint)


def test_affine_checkpoint_rejects_adversarial_legacy_schema_downgrade(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(48)
    train = rng.normal(size=(16, 2)).astype(np.float32)
    checkpoint = tmp_path / "affine-downgraded.ckpt"
    train_model(
        "independent_affine_flow",
        train,
        train[:8],
        _config(
            "posterior_log_noise_affine_flow",
            "log_noise",
            "posterior_mean",
        ),
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["schema_version"] = 1
    del payload["model_contract"]
    torch.save(payload, checkpoint)
    with pytest.raises(
        ValueError,
        match="independent_affine_flow requires checkpoint schema_version 2",
    ):
        load_checkpoint(checkpoint)
