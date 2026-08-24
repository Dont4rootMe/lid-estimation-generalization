from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.pilot import (
    PilotConfigError,
    _selection_coordinate,
    compose_pilot_config,
    validate_pilot_config,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_NAME = (
    "lid-generalization-e8-suite-brownian-schrodinger-bridge-"
    "train-mae-time-selection-seed-137"
)


def _config() -> dict[str, object]:
    composed = compose_pilot_config(("pilot_model=schrodinger_bridge",), root=ROOT)
    return validate_pilot_config(composed)


def test_schrodinger_bridge_hydra_group_seals_the_exact_contract() -> None:
    config = _config()
    assert config["experiment_name"] == EXPECTED_NAME
    assert config["pilot_model"]["family"] == "schrodinger_bridge"
    assert config["pilot_model"]["training"] == {
        "seed": 137,
        "device": "cuda",
        "epochs": 200,
        "batch_size": 512,
        "learning_rate": 0.0002,
        "weight_decay": 0.000001,
        "hidden_dim": 1024,
        "depth": 6,
        "time_embedding_dim": 128,
        "validation_interval": 1,
        "early_stopping_patience": 20,
        "gradient_clip_norm": 1.0,
        "num_workers": 4,
        "deterministic": True,
        "fourier_features": 32,
        "max_condition_frequency": 100.0,
        "dropout": 0.0,
        "normalize": True,
        "normalization_epsilon": 1.0e-8,
        "bridge_construction": "terminal-data-lebesgue-factor-v1",
        "bridge_reference_process": "brownian-motion",
        "bridge_initial_marginal": "gaussian-convolution-of-terminal-data",
        "bridge_terminal_marginal": "dataset-terminal-law",
        "bridge_factor_f": "lebesgue-measure",
        "bridge_factor_g": "dataset-terminal-law",
        "bridge_conditioning": "time-to-go-tau",
        "bridge_diffusivity": 1.0,
        "bridge_terminal_time": 1.0,
        "bridge_tau_min": 0.0001,
        "bridge_tau_max": 1.0,
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("bridge_factor_f", "gaussian-source"),
        ("bridge_factor_g", "empirical-independent-coupling"),
        ("bridge_conditioning", "absolute-time-t"),
        ("bridge_diffusivity", 0.0),
    ],
)
def test_schrodinger_bridge_contract_drift_is_rejected(
    field: str, bad_value: object
) -> None:
    config = deepcopy(_config())
    config["pilot_model"]["training"][field] = bad_value
    with pytest.raises(PilotConfigError, match=field):
        validate_pilot_config(config)


def test_schrodinger_bridge_selection_coordinate_is_time_to_go() -> None:
    import numpy as np

    tau = np.array([0.01, 0.1, 1.0], dtype=np.float64)
    coordinate, name, formula, model_name = _selection_coordinate(
        tau, family="schrodinger_bridge"
    )
    np.testing.assert_array_equal(coordinate, tau)
    assert (name, formula, model_name) == ("tau", "T - t", "tau")
