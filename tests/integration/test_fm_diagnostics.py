from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import experiments.fm_diagnostics as diagnostics_module
from experiments import pilot as pilot_module
from experiments.fm_diagnostics import (
    FMDiagnosticConfigError,
    affine_schedule_point,
    run_fm_diagnostics,
    validate_fm_diagnostic_config,
    validate_fm_diagnostics,
)
from experiments.run_manifest import sha256_path

VARIANTS = (
    "direct_rectified_flow",
    "posterior_rectified_flow",
    "direct_log_noise_affine_flow",
    "posterior_log_noise_affine_flow",
    "direct_vp_trigonometric_flow",
    "posterior_vp_trigonometric_flow",
)
CHECKPOINT_SHA256 = "a" * 64


def _config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "enabled": True,
        "source_split": "train_selection",
        "primary_divergence_backend": "hutchinson",
        "probe_kind": "rademacher",
        "trace_probes": 4,
        "trace_seed": 137,
        "exact_subset_size": 3,
        "exact_subset_seed": 19,
        "oracle_reference_size": 6,
        "oracle_reference_seed": 23,
        "oracle_chunk_size": 2,
        "batch_size": 4,
    }


class _AnalyticPrimitive:
    def __init__(self, variant_id: str) -> None:
        self.variant_id = variant_id
        self.calls: list[tuple[str, int, int]] = []

    def __call__(
        self,
        _trained: object,
        query: np.ndarray,
        scale: float,
        *,
        family: str,
        divergence_backend: str,
        trace_probes: int,
        trace_seed: int,
        batch_size: int,
    ) -> SimpleNamespace:
        assert family == "independent_affine_flow"
        assert trace_seed == 137
        assert batch_size == 4
        self.calls.append((divergence_backend, query.shape[0], trace_probes))
        schedule = diagnostics_module._schedule_name(self.variant_id)
        point = affine_schedule_point(schedule, scale)
        # Mimic a real fp32 predictor and its train-only normalization path.
        channel_point = np.asarray(query, dtype=np.float32).reshape(query.shape[0], -1)
        evaluation_point = np.asarray(
            np.float32(point.alpha) * channel_point, dtype=np.float32
        )
        shrinkage = np.float32(1.0 / (1.0 + scale * scale))
        posterior_mean = np.asarray(shrinkage * channel_point, dtype=np.float32)
        ambient_dim = channel_point.shape[1]
        posterior_divergence = np.full(
            channel_point.shape[0],
            ambient_dim / (point.alpha * (1.0 + scale * scale)),
            dtype=np.float64,
        )
        if divergence_backend == "hutchinson":
            # Deterministic row-local perturbation stands in for a shared
            # Rademacher trace error without changing the posterior field.
            posterior_divergence += 0.01 * np.sign(channel_point[:, 0]).astype(
                np.float64
            )
        velocity = (
            point.alpha_log_derivative + point.log_noise_ratio_derivative
        ) * np.asarray(
            evaluation_point, dtype=np.float64
        ) - point.log_noise_ratio_derivative * point.alpha * np.asarray(
            posterior_mean, dtype=np.float64
        )
        velocity_divergence = (
            (point.alpha_log_derivative + point.log_noise_ratio_derivative)
            * ambient_dim
            - point.log_noise_ratio_derivative * point.alpha * posterior_divergence
        )
        parameterization = (
            "posterior_mean"
            if self.variant_id.startswith("posterior_")
            else "direct_velocity"
        )
        marginal_score = (
            point.alpha * np.asarray(posterior_mean, dtype=np.float64)
            - np.asarray(evaluation_point, dtype=np.float64)
        ) / (point.beta**2)
        marginal_score_divergence = (
            point.alpha * posterior_divergence - ambient_dim
        ) / (point.beta**2)
        return SimpleNamespace(
            velocity=velocity,
            velocity_divergence=velocity_divergence,
            velocity_divergence_from_posterior=velocity_divergence,
            evaluation_point=evaluation_point,
            channel_point=channel_point,
            posterior_mean=posterior_mean,
            posterior_divergence=posterior_divergence,
            marginal_score=marginal_score,
            marginal_score_divergence=marginal_score_divergence,
            variant_id=self.variant_id,
            schedule={
                "rectified": "rectified_linear",
                "log_noise_affine": "log_noise",
                "vp_trigonometric": "vp_trigonometric",
            }[schedule],
            parameterization=parameterization,
            noise_ratio=point.noise_ratio,
            native_time=point.native_coordinate,
            alpha=point.alpha,
            beta=point.beta,
            alpha_derivative=point.alpha_derivative,
            beta_derivative=point.beta_derivative,
            alpha_log_derivative=point.alpha_log_derivative,
            log_noise_ratio_derivative=point.log_noise_ratio_derivative,
            divergence_backend=divergence_backend,
            trace_probe_kind=(
                "rademacher" if divergence_backend == "hutchinson" else "exact"
            ),
            trace_seed=trace_seed if divergence_backend == "hutchinson" else None,
            trace_probes=trace_probes if divergence_backend == "hutchinson" else 0,
            shared_posterior_velocity_probes=(
                divergence_backend == "hutchinson"
                and parameterization == "direct_velocity"
            ),
            primary_trace_field="posterior_mean",
            velocity_divergence_source=(
                "raw_model_trace"
                if parameterization == "direct_velocity"
                else "derived_from_posterior_trace"
            ),
        )


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(2026)
    query = generator.normal(size=(8, 3)).astype(np.float32)
    target = np.full(8, 2.0, dtype=np.float64)
    reference = generator.normal(size=(12, 3)).astype(np.float32)
    return query, target, reference


def _run(tmp_path: Path, variant_id: str) -> tuple[Path, _AnalyticPrimitive]:
    query, target, reference = _inputs()
    primitive = _AnalyticPrimitive(variant_id)
    output = run_fm_diagnostics(
        tmp_path / variant_id,
        variant_id=variant_id,
        trained=SimpleNamespace(checkpoint_sha256=CHECKPOINT_SHA256),
        query=query,
        query_model_space=query,
        target=target,
        oracle_reference_model_space=reference,
        scales=[0.2, 0.6],
        config=_config(),
        primitive_fn=primitive,
    )
    return output, primitive


def test_pilot_hook_runs_and_seals_train_only_diagnostics(tmp_path: Path) -> None:
    query, target, reference = _inputs()
    partition = pilot_module.TrainSelectionPartition(
        fit_indices=np.arange(reference.shape[0], dtype=np.int64),
        selection_indices=np.arange(query.shape[0], dtype=np.int64),
        fit_features=reference,
        selection_features=query,
        selection_target=target,
        record={"protocol": "fixture"},
    )
    trained = SimpleNamespace(
        normalization_mean=np.zeros(query.shape[1], dtype=np.float32),
        normalization_scale=1.0,
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    primitive = _AnalyticPrimitive("posterior_rectified_flow")
    events: list[tuple[str, Mapping[str, Any]]] = []
    assets: list[tuple[str, str]] = []

    class Logger:
        def __call__(self, event: str, payload: Mapping[str, Any]) -> None:
            events.append((event, payload))

        def log_asset(self, path: Path, *, name: str) -> None:
            assets.append((path.name, name))

    cell = tmp_path / "cell"
    cell.mkdir()
    record = pilot_module._run_affine_diagnostics(
        name="e8_gaussian4_pca",
        cell_dir=cell,
        partition=partition,
        trained=trained,
        scales=np.asarray([0.2, 0.6]),
        model={
            "training": {"flow_variant_id": "posterior_rectified_flow"},
            "diagnostics": _config(),
        },
        experiment_name="lid-generalization-e8-suite-fm-fixture",
        log_callback=Logger(),
        diagnostics_fn=run_fm_diagnostics,
        diagnostics_validate_fn=validate_fm_diagnostics,
        primitive_fn=primitive,
    )

    output = cell / "fm_diagnostics"
    assert validate_fm_diagnostics(output) == []
    assert record["path"] == "fm_diagnostics"
    assert record["variant_id"] == "posterior_rectified_flow"
    assert record["source_split"] == "train_selection"
    assert record["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert record["raw_query_sha256"] == diagnostics_module._array_sha256(query)
    assert record["manifest_sha256"] == sha256_path(output / "manifest.json")
    assert [event for event, _ in events] == [
        "dataset.e8_gaussian4_pca.fm_diagnostics.scale",
        "dataset.e8_gaussian4_pca.fm_diagnostics.scale",
        "dataset.e8_gaussian4_pca.fm_diagnostics.completed",
    ]
    assert {name for name, _ in assets} == {
        "summary.json",
        "metadata.json",
        "manifest.json",
    }
    assert not any(
        "validation" in path.name or "test" in path.name for path in output.rglob("*")
    )


def test_outer_cell_binding_rejects_resealed_cross_checkpoint_diagnostics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"the exact trained checkpoint")
    checkpoint_sha = sha256_path(checkpoint)
    scales = np.asarray([0.01, 0.1, 1.0], dtype=np.float64)
    target = np.asarray([2.0, 2.0], dtype=np.float64)
    full = np.asarray([[1.0, 1.5, 2.0], [1.2, 1.7, 2.2]], dtype=np.float64)
    raw_query = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    raw_query_sha = diagnostics_module._array_sha256(raw_query)
    metadata = {
        "checkpoint_sha256": checkpoint_sha,
        "raw_query_sha256": raw_query_sha,
    }
    summary = {
        "checkpoint_sha256": checkpoint_sha,
        "scale_selection": {"partition": {"selection_features_sha256": raw_query_sha}},
    }
    cell_arrays = {
        "scales": scales,
        "train_selection_target": target,
        "train_selection_curve": full,
    }

    def validate(
        *,
        local_metadata: Mapping[str, Any] = metadata,
        local_scales: np.ndarray = scales,
        local_target: np.ndarray = target,
        local_full: np.ndarray = full,
    ) -> list[str]:
        return pilot_module._affine_diagnostic_binding_errors(
            dataset_name="e8_gaussian4_pca",
            metadata=local_metadata,
            diagnostic_scales=local_scales,
            diagnostic_target=local_target,
            diagnostic_full=local_full,
            cell_arrays=cell_arrays,
            summary=summary,
            checkpoint_path=checkpoint,
        )

    assert validate() == []
    assert any(
        "checkpoint.pt" in error
        for error in validate(
            local_metadata={**metadata, "checkpoint_sha256": "0" * 64}
        )
    )
    assert any(
        "train-selection rows" in error
        for error in validate(local_metadata={**metadata, "raw_query_sha256": "1" * 64})
    )
    assert any(
        "scales differ" in error for error in validate(local_scales=scales[::-1])
    )
    assert any("targets differ" in error for error in validate(local_target=target + 1))
    assert any("full curve differs" in error for error in validate(local_full=full + 1))


@pytest.mark.parametrize("variant_id", VARIANTS)
def test_fm_diagnostics_seals_all_variants_and_bounds_expensive_calls(
    tmp_path: Path, variant_id: str
) -> None:
    output, primitive = _run(tmp_path, variant_id)

    assert validate_fm_diagnostics(output) == []
    assert (
        primitive.calls
        == [
            ("hutchinson", 8, 4),
            ("exact", 3, 0),
            ("hutchinson", 3, 4),
        ]
        * 2
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["primary_selection_readout"] == "full"
    for scale in summary["per_scale"]:
        assert set(scale["readouts"]) == {"response", "full", "fm_to_score"}
        assert scale["exact_vs_hutchinson"]["probe_kind"] == "rademacher"
        assert scale["empirical_oracle"]["kind"] == "finite_empirical_reference"
        assert scale["empirical_oracle"]["posterior_weight_ess"]["n"] == 3
    oracle_full = np.load(output / "arrays" / "oracle_full.npy", allow_pickle=False)
    assert oracle_full.shape == (3, 2)
    assert not any(
        "validation" in path.name or "test" in path.name for path in output.rglob("*")
    )


def test_matched_lambda_empirical_oracle_is_schedule_invariant(tmp_path: Path) -> None:
    outputs = [
        _run(tmp_path, variant)[0]
        for variant in (
            "posterior_rectified_flow",
            "posterior_log_noise_affine_flow",
            "posterior_vp_trigonometric_flow",
        )
    ]
    oracle_full = [
        np.load(path / "arrays" / "oracle_full.npy", allow_pickle=False)
        for path in outputs
    ]
    np.testing.assert_allclose(oracle_full[0], oracle_full[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(oracle_full[0], oracle_full[2], rtol=1e-12, atol=1e-12)


def _reseal(directory: Path) -> None:
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for path in (directory / "arrays").glob("*.npy"):
        value = np.load(path, allow_pickle=False)
        metadata["array_sha256"][path.stem] = diagnostics_module._array_sha256(value)
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata_sha256"] = sha256_path(metadata_path)
    manifest["summary_sha256"] = sha256_path(directory / "summary.json")
    manifest["outputs"] = diagnostics_module._output_inventory(directory)
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_validator_rejects_resealed_formula_tampering(tmp_path: Path) -> None:
    output, _ = _run(tmp_path, "posterior_rectified_flow")
    full_path = output / "arrays" / "full.npy"
    full = np.load(full_path, allow_pickle=False)
    full[0, 0] += 7.0
    np.save(full_path, full, allow_pickle=False)
    _reseal(output)

    errors = validate_fm_diagnostics(output)
    assert any("full formula mismatch" in error for error in errors)


def test_validator_rejects_forbidden_evaluation_curve_after_reseal(
    tmp_path: Path,
) -> None:
    output, _ = _run(tmp_path, "posterior_rectified_flow")
    np.save(output / "arrays" / "validation_curve.npy", np.ones((2, 2)))
    _reseal(output)

    errors = validate_fm_diagnostics(output)
    assert any("forbidden validation/test" in error for error in errors)


def test_config_is_explicit_and_train_selection_only() -> None:
    config = _config()
    del config["oracle_chunk_size"]
    with pytest.raises(FMDiagnosticConfigError, match="missing"):
        validate_fm_diagnostic_config(config)
    config = _config()
    config["source_split"] = "validation"
    with pytest.raises(FMDiagnosticConfigError, match="train_selection"):
        validate_fm_diagnostic_config(config)


@pytest.mark.parametrize("checkpoint_sha256", (None, "A" * 64, "abc"))
def test_run_requires_canonical_checkpoint_provenance(
    tmp_path: Path,
    checkpoint_sha256: str | None,
) -> None:
    query, target, reference = _inputs()
    with pytest.raises(FMDiagnosticConfigError, match="checkpoint_sha256"):
        run_fm_diagnostics(
            tmp_path / "invalid-provenance",
            variant_id="direct_rectified_flow",
            trained=SimpleNamespace(checkpoint_sha256=checkpoint_sha256),
            query=query,
            query_model_space=query,
            target=target,
            oracle_reference_model_space=reference,
            scales=[0.2, 0.6],
            config=_config(),
            primitive_fn=_AnalyticPrimitive("direct_rectified_flow"),
        )


def test_validator_rejects_resealed_malformed_input_provenance(tmp_path: Path) -> None:
    output, _ = _run(tmp_path, "posterior_rectified_flow")
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["raw_query_sha256"] = "not-a-sha"
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _reseal(output)

    errors = validate_fm_diagnostics(output)
    assert "invalid FM diagnostic raw_query_sha256" in errors


@pytest.mark.parametrize(
    "variant_id,schedule,parameterization",
    (
        ("direct_rectified_flow", "rectified_linear", "direct_velocity"),
        ("posterior_rectified_flow", "rectified_linear", "posterior_mean"),
        (
            "direct_vp_trigonometric_flow",
            "vp_trigonometric",
            "direct_velocity",
        ),
        (
            "posterior_vp_trigonometric_flow",
            "vp_trigonometric",
            "posterior_mean",
        ),
    ),
)
def test_real_fp32_endpoint_score_conversion_uses_predictor_arithmetic(
    tmp_path: Path,
    variant_id: str,
    schedule: str,
    parameterization: str,
) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from models.affine_flow import (
        AffineFlowSpec,
        affine_schedule_state,
        posterior_to_velocity,
    )
    from models.training import predict_affine_primitives

    spec = AffineFlowSpec.from_mapping(
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

    class AnalyticFP32Field(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.config = SimpleNamespace(ambient_dim=3)
            self._lid_family = "independent_affine_flow"
            self._lid_affine_spec = spec
            self.checkpoint_sha256 = CHECKPOINT_SHA256

        def forward(self, inputs, condition):  # type: ignore[no-untyped-def]
            state = affine_schedule_state(torch.exp(condition), spec.schedule)
            coefficient = state.alpha / (state.alpha.square() + state.beta.square())
            posterior = coefficient[:, None] * inputs + self.anchor * 0.0
            if spec.parameterization == "posterior_mean":
                return posterior
            return posterior_to_velocity(posterior, inputs, state)

    query, target, reference = _inputs()
    config = _config()
    config["exact_subset_size"] = 2
    config["oracle_reference_size"] = 4
    output = run_fm_diagnostics(
        tmp_path / "real-fp32",
        variant_id=spec.variant_id,
        trained=AnalyticFP32Field(),
        query=query,
        query_model_space=query,
        target=target,
        oracle_reference_model_space=reference,
        scales=[0.01, 0.02],
        config=config,
        primitive_fn=predict_affine_primitives,
    )

    assert validate_fm_diagnostics(output) == []
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["compute_dtype"] == "float32"
    residual = np.load(
        output / "arrays" / "evaluation_point_residual_norm.npy",
        allow_pickle=False,
    )
    assert residual.max() > 0.0
    assert residual.max() < 1.0e-5
    actual = np.load(output / "arrays" / "fm_to_score.npy", allow_pickle=False)
    ideal = np.load(output / "arrays" / "fm_to_score_ideal.npy", allow_pickle=False)
    full = np.load(output / "arrays" / "full.npy", allow_pickle=False)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(ideal, full, rtol=1e-11, atol=1e-11)


def test_posterior_vp_conversion_bound_accounts_for_operand_cancellation(
    tmp_path: Path,
) -> None:
    """A one-ulp kernel difference is scaled by the cancelling operands.

    This reproduces the production posterior-VP failure at lambda=.3162 and
    ambient dimension 784: two roughly 1e3 affine terms cancel to a velocity
    near one, so one float32 ulp of an operand is much larger than a
    result-relative allowance.  The operand-conditioned conversion must pass,
    while a genuine field mismatch must still fail.
    """

    ambient_dim = 784
    point = affine_schedule_point("vp_trigonometric", 0.3162)
    state = diagnostics_module._numeric_schedule_scalars(point, compute_dtype="float32")
    evaluation_point = np.full((1, ambient_dim), 210.0, dtype=np.float32)
    point_term = state["beta_log_derivative"] * evaluation_point
    coefficient = state["alpha_log_noise_derivative"]
    posterior_mean = np.asarray(
        (point_term - np.float32(1.0)) / coefficient,
        dtype=np.float32,
    )
    cpu_velocity = np.asarray(
        point_term - coefficient * posterior_mean,
        dtype=np.float32,
    )
    gpu_like_point_term = np.nextafter(
        point_term,
        np.full_like(point_term, np.inf),
    )
    gpu_like_velocity = np.asarray(
        gpu_like_point_term - coefficient * posterior_mean,
        dtype=np.float32,
    )
    operand_ulp = float(gpu_like_point_term[0, 0] - point_term[0, 0])
    assert operand_ulp == pytest.approx(0.0001220703125)
    assert float(gpu_like_velocity[0, 0] - cpu_velocity[0, 0]) == pytest.approx(
        operand_ulp
    )
    with pytest.raises(ValueError, match="unconditioned control exceeds"):
        diagnostics_module._require_roundoff_consistency(
            np.asarray(gpu_like_velocity, dtype=np.float64),
            np.asarray(cpu_velocity, dtype=np.float64),
            compute_dtype="float32",
            label="unconditioned control",
        )

    posterior_divergence = np.asarray([float(ambient_dim)], dtype=np.float32)
    alpha = state["alpha"]
    beta = state["beta"]
    score = np.asarray(
        (alpha * posterior_mean - evaluation_point) / np.square(beta),
        dtype=np.float32,
    )
    score_divergence = np.asarray(
        (
            alpha * posterior_divergence
            - np.asarray(float(ambient_dim), dtype=np.float32)
        )
        / np.square(beta),
        dtype=np.float32,
    )
    velocity_divergence = np.asarray(
        [
            (point.alpha_log_derivative + point.log_noise_ratio_derivative)
            * ambient_dim
            - point.log_noise_ratio_derivative
            * point.alpha
            * float(posterior_divergence[0])
        ],
        dtype=np.float64,
    )

    def primitives(velocity: np.ndarray) -> diagnostics_module.AffinePrimitiveBatch:
        return diagnostics_module.AffinePrimitiveBatch(
            velocity=np.asarray(velocity, dtype=np.float64),
            velocity_divergence=velocity_divergence,
            velocity_divergence_from_posterior=velocity_divergence,
            evaluation_point=np.asarray(evaluation_point, dtype=np.float64),
            posterior_mean=np.asarray(posterior_mean, dtype=np.float64),
            posterior_divergence=np.asarray(posterior_divergence, dtype=np.float64),
            score=np.asarray(score, dtype=np.float64),
            score_divergence=np.asarray(score_divergence, dtype=np.float64),
        )

    query_model_space = np.asarray(evaluation_point, dtype=np.float64) / point.alpha
    derived = diagnostics_module._derived_components(
        primitives(gpu_like_velocity),
        query_model_space,
        point,
        compute_dtype="float32",
        parameterization="posterior_mean",
    )
    np.testing.assert_allclose(
        derived["velocity_posterior_predictor_residual_norm"],
        [np.sqrt(ambient_dim) * operand_ulp],
        rtol=1e-12,
        atol=1e-12,
    )

    corrupted_velocity = np.asarray(gpu_like_velocity + np.float32(0.1))
    with pytest.raises(
        ValueError,
        match="velocity reconstructed from posterior exceeds",
    ):
        diagnostics_module._derived_components(
            primitives(corrupted_velocity),
            query_model_space,
            point,
            compute_dtype="float32",
            parameterization="posterior_mean",
        )

    class Float32Marker:
        def parameters(self):  # type: ignore[no-untyped-def]
            return iter((SimpleNamespace(dtype="torch.float32"),))

    def cancellation_primitive(
        _trained: object,
        raw_query: np.ndarray,
        scale: float,
        *,
        family: str,
        divergence_backend: str,
        trace_probes: int,
        trace_seed: int,
        batch_size: int,
    ) -> SimpleNamespace:
        assert family == "independent_affine_flow"
        assert trace_seed == 137
        assert batch_size == 1
        local_point = affine_schedule_point("vp_trigonometric", scale)
        local_state = diagnostics_module._numeric_schedule_scalars(
            local_point, compute_dtype="float32"
        )
        channel_point = np.asarray(raw_query, dtype=np.float32).reshape(
            raw_query.shape[0], -1
        )
        local_evaluation_point = np.asarray(
            local_state["alpha"] * channel_point,
            dtype=np.float32,
        )
        local_point_term = local_state["beta_log_derivative"] * local_evaluation_point
        local_coefficient = local_state["alpha_log_noise_derivative"]
        local_posterior = np.asarray(
            (local_point_term - np.float32(1.0)) / local_coefficient,
            dtype=np.float32,
        )
        gpu_like_term = np.nextafter(
            local_point_term,
            np.full_like(local_point_term, np.inf),
        )
        local_velocity = np.asarray(
            gpu_like_term - local_coefficient * local_posterior,
            dtype=np.float32,
        )
        local_posterior_divergence = np.full(
            channel_point.shape[0],
            float(channel_point.shape[1]),
            dtype=np.float32,
        )
        local_velocity_divergence = (
            local_point.alpha_log_derivative + local_point.log_noise_ratio_derivative
        ) * channel_point.shape[
            1
        ] - local_point.log_noise_ratio_derivative * local_point.alpha * np.asarray(
            local_posterior_divergence, dtype=np.float64
        )
        local_score = np.asarray(
            (local_state["alpha"] * local_posterior - local_evaluation_point)
            / np.square(local_state["beta"]),
            dtype=np.float32,
        )
        local_score_divergence = np.asarray(
            (
                local_state["alpha"] * local_posterior_divergence
                - np.asarray(float(channel_point.shape[1]), dtype=np.float32)
            )
            / np.square(local_state["beta"]),
            dtype=np.float32,
        )
        return SimpleNamespace(
            velocity=local_velocity,
            velocity_divergence=local_velocity_divergence,
            velocity_divergence_from_posterior=local_velocity_divergence,
            evaluation_point=local_evaluation_point,
            channel_point=channel_point,
            posterior_mean=local_posterior,
            posterior_divergence=local_posterior_divergence,
            marginal_score=local_score,
            marginal_score_divergence=local_score_divergence,
            variant_id="posterior_vp_trigonometric_flow",
            schedule="vp_trigonometric",
            parameterization="posterior_mean",
            noise_ratio=local_point.noise_ratio,
            native_time=local_point.native_coordinate,
            alpha=local_point.alpha,
            beta=local_point.beta,
            alpha_derivative=local_point.alpha_derivative,
            beta_derivative=local_point.beta_derivative,
            alpha_log_derivative=local_point.alpha_log_derivative,
            log_noise_ratio_derivative=local_point.log_noise_ratio_derivative,
            divergence_backend=divergence_backend,
            trace_probe_kind=(
                "rademacher" if divergence_backend == "hutchinson" else "exact"
            ),
            trace_seed=trace_seed if divergence_backend == "hutchinson" else None,
            trace_probes=trace_probes if divergence_backend == "hutchinson" else 0,
            shared_posterior_velocity_probes=False,
            primary_trace_field="posterior_mean",
            velocity_divergence_source="derived_from_posterior_trace",
        )

    sealed_query = np.full((2, ambient_dim), 220.0, dtype=np.float32)
    sealed_config = _config()
    sealed_config.update(
        exact_subset_size=1,
        oracle_reference_size=2,
        oracle_chunk_size=1,
        batch_size=1,
    )
    output = run_fm_diagnostics(
        tmp_path / "sealed-posterior-vp-cancellation",
        variant_id="posterior_vp_trigonometric_flow",
        trained=SimpleNamespace(
            model=Float32Marker(), checkpoint_sha256=CHECKPOINT_SHA256
        ),
        query=sealed_query,
        query_model_space=sealed_query,
        target=np.full(sealed_query.shape[0], 5.0, dtype=np.float64),
        oracle_reference_model_space=np.vstack(
            (sealed_query, sealed_query + np.float32(0.25))
        ),
        scales=[0.3162, 0.5],
        config=sealed_config,
        primitive_fn=cancellation_primitive,
    )
    assert validate_fm_diagnostics(output) == []
    sealed_residual = np.load(
        output / "arrays" / "velocity_posterior_predictor_residual_norm.npy",
        allow_pickle=False,
    )
    assert sealed_residual.shape == (2, 2)
    assert np.all(sealed_residual > 0.0)


def test_real_trained_direct_rectified_field_identity_uses_predictor_arithmetic(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from models.training import (
        TrainingConfig,
        predict_affine_primitives,
        train_model,
    )

    generator = np.random.default_rng(9107)
    fit = generator.normal(size=(24, 6)).astype(np.float32)
    selection = generator.normal(size=(12, 6)).astype(np.float32)
    trained = train_model(
        "independent_affine_flow",
        fit,
        selection,
        TrainingConfig(
            seed=41,
            device="cpu",
            epochs=2,
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
            flow_variant_id="direct_rectified_flow",
            flow_schedule="rectified_linear",
            flow_parameterization="direct_velocity",
            flow_conditioning="log_noise_ratio",
            flow_scale_sampling="log_uniform_noise_ratio",
            flow_loss_weighting="posterior_bias_equivalent",
            flow_noise_ratio_min=0.01,
            flow_noise_ratio_max=1.0,
        ),
        tmp_path / "trained-direct-rf.pt",
    )
    selection_model_space = pilot_module._features_in_model_space(
        trained,
        selection,
        label="selection",
    )
    fit_model_space = pilot_module._features_in_model_space(
        trained,
        fit,
        label="fit",
    )
    output = run_fm_diagnostics(
        tmp_path / "trained-direct-rf-diagnostics",
        variant_id="direct_rectified_flow",
        trained=trained,
        query=selection,
        query_model_space=selection_model_space,
        target=np.full(selection.shape[0], 2.0, dtype=np.float64),
        oracle_reference_model_space=fit_model_space,
        scales=[0.01, 0.1, 1.0],
        config=_config(),
        primitive_fn=predict_affine_primitives,
    )

    assert validate_fm_diagnostics(output) == []
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint_sha256"] == trained.checkpoint_sha256
    predictor_residual = np.load(
        output / "arrays" / "posterior_velocity_predictor_residual_norm.npy",
        allow_pickle=False,
    )
    ideal_residual = np.load(
        output / "arrays" / "posterior_velocity_ideal_residual_norm.npy",
        allow_pickle=False,
    )
    assert predictor_residual.max() <= 1.0e-5
    assert ideal_residual[:, 0].max() > predictor_residual[:, 0].max()
