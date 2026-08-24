from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from experiments import pilot
from experiments.pilot import (
    PilotConfigError,
    compose_pilot_config,
    validate_pilot_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

VARIANTS = {
    "direct_rectified_flow": (
        "rectified_linear",
        "direct_velocity",
        "fm-rectified-direct-velocity",
    ),
    "posterior_rectified_flow": (
        "rectified_linear",
        "posterior_mean",
        "fm-rectified-posterior-mean",
    ),
    "direct_log_noise_affine_flow": (
        "log_noise",
        "direct_velocity",
        "fm-log-noise-direct-velocity",
    ),
    "posterior_log_noise_affine_flow": (
        "log_noise",
        "posterior_mean",
        "fm-log-noise-posterior-mean",
    ),
    "direct_vp_trigonometric_flow": (
        "vp_trigonometric",
        "direct_velocity",
        "fm-vp-trigonometric-direct-velocity",
    ),
    "posterior_vp_trigonometric_flow": (
        "vp_trigonometric",
        "posterior_mean",
        "fm-vp-trigonometric-posterior-mean",
    ),
}
COMMON_LAMBDA_GRID = [
    0.01,
    0.0178,
    0.0316,
    0.0562,
    0.1,
    0.1778,
    0.3162,
    0.5623,
    1.0,
]
DIAGNOSTIC_FIELDS = {
    "schema_version",
    "enabled",
    "source_split",
    "primary_divergence_backend",
    "probe_kind",
    "trace_probes",
    "trace_seed",
    "exact_subset_size",
    "exact_subset_seed",
    "oracle_reference_size",
    "oracle_reference_seed",
    "oracle_chunk_size",
    "batch_size",
}


@pytest.mark.parametrize(
    ("variant_id", "identity"),
    VARIANTS.items(),
)
def test_affine_fm_hydra_group_seals_factorial_identity(
    variant_id: str, identity: tuple[str, str, str]
) -> None:
    schedule, parameterization, readable_name = identity
    resolved = validate_pilot_config(
        compose_pilot_config([f"pilot_model={variant_id}"], root=REPOSITORY_ROOT)
    )
    model = resolved["pilot_model"]
    training = model["training"]

    assert model["name"] == variant_id
    assert model["family"] == "independent_affine_flow"
    assert model["readout"] == "full"
    assert model["selection_prefer"] == "smaller"
    assert model["derivative_backend"] == "hutchinson"
    assert model["trace_probes"] == 16
    assert list(model["scales"]) == COMMON_LAMBDA_GRID
    assert training["flow_variant_id"] == variant_id
    assert training["flow_schedule"] == schedule
    assert training["flow_parameterization"] == parameterization
    assert training["flow_conditioning"] == "log_noise_ratio"
    assert training["flow_scale_sampling"] == "log_uniform_noise_ratio"
    assert training["flow_loss_weighting"] == "posterior_bias_equivalent"
    assert training["flow_noise_ratio_min"] == 0.01
    assert training["flow_noise_ratio_max"] == 1.0
    assert not {
        "sigma_min",
        "sigma_max",
        "time_min",
        "time_max",
        "epsilon_min",
        "bridge_tau_min",
    } & set(training)

    experiment_name = resolved["experiment_name"]
    assert readable_name in experiment_name
    assert experiment_name.endswith(
        "-all-readouts-debug-train-mae-lambda-selection-seed-137"
    )
    assert "ent-block" not in experiment_name
    assert "eval" not in experiment_name
    assert resolved["logging"]["experiment_name"] == experiment_name

    diagnostics = model["diagnostics"]
    assert set(diagnostics) == DIAGNOSTIC_FIELDS
    assert diagnostics == {
        "schema_version": 1,
        "enabled": True,
        "source_split": "train_selection",
        "primary_divergence_backend": "hutchinson",
        "probe_kind": "rademacher",
        "trace_probes": 16,
        "trace_seed": 137,
        "exact_subset_size": 32,
        "exact_subset_seed": 137,
        "oracle_reference_size": 4096,
        "oracle_reference_seed": 137,
        "oracle_chunk_size": 1024,
        "batch_size": 128,
    }


def test_affine_fm_seed_override_reaches_name_training_and_diagnostics() -> None:
    resolved = validate_pilot_config(
        compose_pilot_config(
            ["pilot_model=posterior_vp_trigonometric_flow", "seed=23"],
            root=REPOSITORY_ROOT,
        )
    )
    assert resolved["experiment_name"].endswith("-seed-23")
    assert resolved["pilot_model"]["training"]["seed"] == 23
    diagnostics = resolved["pilot_model"]["diagnostics"]
    assert diagnostics["trace_seed"] == 23
    assert diagnostics["exact_subset_seed"] == 23
    assert diagnostics["oracle_reference_seed"] == 23


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["pilot_model"]["training"].__setitem__(
                "flow_parameterization", "posterior_mean"
            ),
            "flow_parameterization must be exactly",
        ),
        (
            lambda value: value["pilot_model"]["training"].__setitem__(
                "flow_conditioning", "native_time"
            ),
            "flow_conditioning must be exactly",
        ),
        (
            lambda value: value["pilot_model"]["training"].__setitem__(
                "time_min", 0.01
            ),
            "unknown pilot_model.training",
        ),
        (
            lambda value: value["pilot_model"]["scales"].__setitem__(0, 0.001),
            "exact common lambda grid",
        ),
        (
            lambda value: value["pilot_model"]["diagnostics"].pop("oracle_chunk_size"),
            "missing pilot_model.diagnostics",
        ),
        (
            lambda value: value["pilot_model"]["diagnostics"].__setitem__(
                "source_split", "validation"
            ),
            "source_split must be exactly",
        ),
        (
            lambda value: value["pilot_model"]["diagnostics"].__setitem__(
                "exact_subset_size", 16
            ),
            "exact_subset_size must be exactly 32",
        ),
        (
            lambda value: value["pilot_model"]["diagnostics"].__setitem__(
                "oracle_reference_seed", 99
            ),
            "oracle_reference_seed must equal evaluation.trace_seed",
        ),
    ],
)
def test_affine_fm_config_rejects_identity_and_debug_drift(
    mutation: Any, message: str
) -> None:
    value = OmegaConf.create(
        validate_pilot_config(
            compose_pilot_config(
                ["pilot_model=direct_rectified_flow"], root=REPOSITORY_ROOT
            )
        )
    )
    mutation(value)
    with pytest.raises(PilotConfigError, match=message):
        validate_pilot_config(value)


def test_affine_selection_is_lambda_and_reports_native_schedule_coordinate() -> None:
    scales = np.asarray(COMMON_LAMBDA_GRID, dtype=np.float64)
    coordinates, name, formula, model_name = pilot._selection_coordinate(
        scales, family="independent_affine_flow"
    )
    assert np.array_equal(coordinates, scales)
    assert (name, formula, model_name) == ("lambda", "beta / alpha", "lambda")

    raw = {
        "schema_version": 1,
        "selected_index": 0,
        "selected_scale": 0.01,
    }
    for variant_id, (schedule, parameterization, _) in VARIANTS.items():
        reported = pilot._reported_selection_diagnostics(
            deepcopy(raw),
            scales=scales,
            coordinates=coordinates,
            selected_index=0,
            coordinate_name=name,
            coordinate_formula=formula,
            model_scale_name=model_name,
            coordinate_prefer="smaller",
            model_scale_prefer="smaller",
            model_training={
                "flow_variant_id": variant_id,
                "flow_schedule": schedule,
                "flow_parameterization": parameterization,
            },
        )
        assert reported["selected_scale"] == reported["selected_lambda"] == 0.01
        assert reported["affine_flow"]["variant_id"] == variant_id
        assert reported["affine_flow"]["scale_semantics"] == (
            "noise_ratio_lambda=beta/alpha"
        )
        assert np.isfinite(
            reported["affine_flow"]["selected_native_coordinate"]["value"]
        )


def test_affine_checkpoint_contract_must_match_hydra_variant() -> None:
    resolved = validate_pilot_config(
        compose_pilot_config(
            ["pilot_model=posterior_log_noise_affine_flow"], root=REPOSITORY_ROOT
        )
    )
    training = resolved["pilot_model"]["training"]

    class Result:
        def __init__(self, contract: dict[str, Any]) -> None:
            self.model_contract = contract

    contract = {
        "family": "independent_affine_flow",
        "scale_semantics": "noise_ratio_lambda=beta/alpha",
        "variant_id": "posterior_log_noise_affine_flow",
        "schedule": "log_noise",
        "parameterization": "posterior_mean",
        "conditioning": "log_noise_ratio",
        "scale_sampling": "log_uniform_noise_ratio",
        "loss_weighting": "posterior_bias_equivalent",
        "noise_ratio_min": 0.01,
        "noise_ratio_max": 1.0,
        "readouts": ["response", "full", "fm_to_score"],
    }

    pilot._validate_affine_training_result(Result(contract), training=training)
    mismatched = {**contract, "schedule": "rectified_linear"}
    with pytest.raises(RuntimeError, match="checkpoint contract differs"):
        pilot._validate_affine_training_result(Result(mismatched), training=training)


def test_frozen_affine_readouts_use_one_lambda_and_no_evaluation_curves() -> None:
    calls: list[tuple[str, float, int]] = []

    def predict(_trained, query, scale, *, readout, **_kwargs):
        calls.append((readout, scale, len(query)))
        offset = {"response": 1.0, "full": 2.0, "fm_to_score": 3.0}[readout]
        return np.full(len(query), offset, dtype=np.float64)

    model = {"readout": "full"}
    evaluation = {
        "divergence_backend": "hutchinson",
        "trace_probes": 16,
        "trace_seed": 137,
        "batch_size": 128,
    }
    queries = {
        "train_selection": np.zeros((4, 2)),
        "validation": np.zeros((3, 2)),
        "test": np.zeros((2, 2)),
    }
    targets = {name: np.full(len(query), 2.0) for name, query in queries.items()}
    primary = {name: np.full(len(query), 2.0) for name, query in queries.items()}

    predictions, summary = pilot._frozen_affine_readouts(
        predict_fn=predict,
        trained=object(),
        selected_lambda=0.1,
        model=model,
        evaluation=evaluation,
        split_queries=queries,
        split_targets=targets,
        primary_predictions=primary,
    )
    assert calls == [
        ("response", 0.1, 4),
        ("fm_to_score", 0.1, 4),
        ("response", 0.1, 3),
        ("fm_to_score", 0.1, 3),
        ("response", 0.1, 2),
        ("fm_to_score", 0.1, 2),
    ]
    assert set(predictions) == {"train_selection", "validation", "test"}
    assert all(
        set(value) == set(pilot._AFFINE_READOUTS) for value in predictions.values()
    )
    assert summary["selected_lambda"] == 0.1
    assert summary["primary_readout"] == "full"
    assert summary["readouts"]["full"]["validation"]["mae"] == 0.0
    assert summary["retrospective_validation_curves_saved"] is False
    assert summary["retrospective_test_curves_saved"] is False


def test_frozen_affine_readouts_share_one_primitive_trace_per_split() -> None:
    primitive_calls: list[tuple[float, int]] = []

    class Primitives:
        def __init__(self, n_samples: int) -> None:
            self.posterior_mean = np.full((n_samples, 2), 0.5)
            self.channel_point = np.zeros((n_samples, 2))
            self.posterior_divergence = np.full(n_samples, 2.0)
            self.marginal_score = np.full((n_samples, 2), 0.25)
            self.marginal_score_divergence = np.full(n_samples, -1.0)
            self.alpha = 0.8
            self.beta = 0.4
            self.noise_ratio = 0.5

    def primitives(_trained, query, scale, **_kwargs):
        primitive_calls.append((scale, len(query)))
        return Primitives(len(query))

    def forbidden_predict(*_args, **_kwargs):
        raise AssertionError("readout-specific predictor must not be called")

    queries = {
        "train_selection": np.zeros((4, 2)),
        "validation": np.zeros((3, 2)),
        "test": np.zeros((2, 2)),
    }
    targets = {name: np.zeros(len(query)) for name, query in queries.items()}
    expected_train = pilot._readouts_from_affine_primitives(Primitives(4))["full"]
    predictions, summary = pilot._frozen_affine_readouts(
        predict_fn=forbidden_predict,
        trained=object(),
        selected_lambda=0.5,
        model={"readout": "full"},
        evaluation={
            "divergence_backend": "hutchinson",
            "trace_probes": 16,
            "trace_seed": 137,
            "batch_size": 128,
        },
        split_queries=queries,
        split_targets=targets,
        primary_predictions={"train_selection": expected_train},
        primitive_fn=primitives,
    )
    assert primitive_calls == [(0.5, 4), (0.5, 3), (0.5, 2)]
    assert set(predictions["validation"]) == {
        "response",
        "full",
        "fm_to_score",
    }
    assert summary["readouts"]["response"]["test"]["n"] == 2


def test_diagnostic_model_space_conversion_matches_predictor_fp32_semantics() -> None:
    class Trained:
        normalization_mean = torch.tensor([0.2, -0.3], dtype=torch.float32)
        normalization_scale = 0.7

    raw = np.asarray(
        [[0.123456789, -0.987654321], [1.23456789, 2.34567891]],
        dtype=np.float64,
    )
    expected_raw = torch.as_tensor(raw).to(dtype=torch.float32)
    expected = (
        (expected_raw - Trained.normalization_mean.reshape(1, -1))
        / Trained.normalization_scale
    ).numpy()
    actual = pilot._features_in_model_space(Trained(), raw, label="fixture")
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)
