from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf
import pytest
import yaml

import experiments.pilot as pilot_module
from experiments.pilot import (
    EXPERIMENT_NAME,
    PILOT_DATASETS,
    PROJECT_NAME,
    PilotConfigError,
    compose_pilot_config,
    run_pilot,
    validate_pilot_config,
    validate_pilot_experiment,
)
from models.learned import load_artifact_registry, verify_model_artifacts
from utils.provenance import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _FakeTrainingResult:
    model: object
    family: str
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str
    metrics: dict[str, float]


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, Any]]] = []
        self.assets: list[tuple[Path, str]] = []

    def __call__(self, event: str, payload: Mapping[str, Any]) -> None:
        self.events.append((event, payload))

    def log_asset(self, path: Path, *, name: str) -> None:
        assert path.is_file()
        self.assets.append((path, name))


def _write_fixture_dataset(root: Path, registry_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for dataset_index, name in enumerate(PILOT_DATASETS, start=2):
        rows.append(
            {
                "name": name,
                "representation": "dataset",
                "available_representations": ["dataset"],
                "required_artifacts": ["dataset", "lid"],
                "expected_samples": {"train": 12, "val": 6, "test": 6},
                "expected_shapes": {"dataset": [1, 2, 2]},
                "expected_lid": dataset_index,
                "official": False,
            }
        )
        for split, n_samples in (("train", 12), ("val", 6), ("test", 6)):
            split_dir = root / name / split
            split_dir.mkdir(parents=True)
            # Feature mean identifies the fixture to the fake model without ever
            # passing a LID target through the trainer or selector interface.
            features = np.full(
                (n_samples, 1, 2, 2), float(dataset_index), dtype=np.float32
            )
            features[:, 0, 0, 0] += np.linspace(0.0, 0.01, n_samples)
            np.save(split_dir / "dataset.npy", features, allow_pickle=False)
            np.save(
                split_dir / "lid.npy",
                np.full(n_samples, dataset_index, dtype=np.float32),
                allow_pickle=False,
            )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmark_id": "pilot-test-fixture",
                "datasets": rows,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _fixture_config(tmp_path: Path, *, family: str = "diffusion"):
    data_root = tmp_path / "benchmarks"
    registry = tmp_path / "registry.yaml"
    _write_fixture_dataset(data_root, registry)
    group = "diffusion" if family == "diffusion" else "rectified_flow"
    config = compose_pilot_config(
        [
            f"pilot_model={group}",
            f"data.root={data_root}",
            f"data.registry={registry}",
            "logging.backend=none",
            "pilot_model.training.device=cpu",
        ],
        root=REPOSITORY_ROOT,
    )
    return config


def _fake_train(
    family: str,
    train: np.ndarray,
    validation: np.ndarray,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    log_callback=None,
) -> _FakeTrainingResult:
    assert train.ndim == validation.ndim == 2
    assert train.shape[1] == validation.shape[1] == 4
    value = float(np.round(train.mean()))
    checkpoint_path.write_bytes(f"{family}:{value}".encode("utf-8"))
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if log_callback is not None:
        log_callback(
            {"family": family, "epoch": 1, "validation_loss": 0.25}
        )
    return _FakeTrainingResult(
        model={"value": value},
        family="gaussian_diffusion" if family == "diffusion" else family,
        history=[{"epoch": 1.0, "training_loss": 0.5, "validation_loss": 0.25}],
        best_epoch=1,
        best_validation_loss=0.25,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=digest,
        metrics={"training_loss": 0.5},
    )


def _fake_predict(
    result: _FakeTrainingResult,
    query: np.ndarray,
    scale: float,
    **_: Any,
) -> np.ndarray:
    # A smooth target-independent curve with a plateau around the middle of the
    # configured scale grid.  ``query`` is used only for the output cardinality.
    grid_position = abs(np.log(scale) - np.log(0.18))
    offset = 0.02 * grid_position**2
    return np.full(query.shape[0], result.model["value"] + offset)


@pytest.mark.parametrize("family", ["diffusion", "rectified_flow"])
def test_pilot_trains_three_models_and_seals_all_metrics(
    tmp_path: Path, family: str
) -> None:
    config = _fixture_config(tmp_path, family=family)
    logger = _RecordingLogger()
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
        log_callback=logger,
    )

    assert validate_pilot_experiment(run_dir) == []
    completed_events = {
        event for event, _ in logger.events if event.endswith(".completed")
    }
    assert completed_events == {
        f"dataset.{name}.completed" for name in PILOT_DATASETS
    } | {"experiment.completed"}
    training_events = [
        (event, payload)
        for event, payload in logger.events
        if event.endswith(".training.epoch")
    ]
    assert {event for event, _ in training_events} == {
        f"dataset.{name}.training.epoch" for name in PILOT_DATASETS
    }
    assert all(
        payload["step"] == payload["epoch"] == 1
        for _, payload in training_events
    )
    assert all(payload["project"] == PROJECT_NAME for _, payload in logger.events)
    assert all(
        payload["experiment_name"] == EXPERIMENT_NAME
        for _, payload in logger.events
    )
    assert {path.name for path, _ in logger.assets} == {
        "summary.json",
        "manifest.json",
        "resolved_config.yaml",
    }
    completion = next(
        payload for event, payload in logger.events if event == "experiment.completed"
    )
    assert completion["shared_filesystem_run_dir"] == str(run_dir)
    assert completion["uploaded_assets"] == {
        "summary.json": True,
        "manifest.json": True,
        "resolved_config.yaml": True,
    }
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert tuple(summary["datasets"]) == PILOT_DATASETS
    assert summary["selection_uses_lid_targets"] is False
    assert set(summary["macro_validation"]) >= {"mean_mae", "mean_rmse"}
    assert set(summary["macro_test"]) >= {"mean_mae", "mean_rmse"}

    registry = yaml.safe_load(
        (run_dir / "artifact_registry.yaml").read_text(encoding="utf-8")
    )
    assert set(registry["artifacts"]) == {
        f"{name}/dataset" for name in PILOT_DATASETS
    }
    learned_model = {
        "name": config.pilot_model.name,
        "family": (
            "gaussian_diffusion" if family == "diffusion" else family
        ),
        "seed": int(config.seed),
        "artifact_registry": str(run_dir / "artifact_registry.yaml"),
        "artifact_registry_sha256": sha256_file(
            run_dir / "artifact_registry.yaml"
        ),
    }
    loaded_registry = load_artifact_registry(root=tmp_path, model=learned_model)
    for name in PILOT_DATASETS:
        cell = run_dir / "datasets" / name
        for filename in (
            "checkpoint.pt",
            "training.yaml",
            "training_history.json",
            "scales.npy",
            "validation_curve.npy",
            "test_curve.npy",
            "validation_prediction.npy",
            "test_prediction.npy",
            "validation_target.npy",
            "test_target.npy",
            "summary.json",
        ):
            assert (cell / filename).is_file()
        cell_summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
        assert cell_summary["selection_uses_lid_targets"] is False
        assert cell_summary["scale_selection"]["uses_ground_truth"] is False
        assert cell_summary["validation"]["mae"] < 0.2
        assert cell_summary["test"]["mae"] < 0.2
        verified = verify_model_artifacts(
            registry=loaded_registry,
            model=learned_model,
            dataset_name=name,
            representation="dataset",
            training_dataset_sha256=cell_summary["training_dataset_sha256"],
            preprocessing_sha256=cell_summary["preprocessing_sha256"],
        )
        assert verified.checkpoint_sha256 == cell_summary["checkpoint_sha256"]


def test_scale_selection_precedes_any_label_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    selected = False
    original_select = pilot_module.select_stable_scale
    original_metric = pilot_module.known_lid_metrics

    def guarded_select(*args, **kwargs):
        nonlocal selected
        result = original_select(*args, **kwargs)
        selected = True
        return result

    def guarded_metric(*args, **kwargs):
        assert selected, "LID target metric was accessed before scale selection"
        return original_metric(*args, **kwargs)

    monkeypatch.setattr(pilot_module, "select_stable_scale", guarded_select)
    monkeypatch.setattr(pilot_module, "known_lid_metrics", guarded_metric)
    run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
    )


def test_rectified_flow_selects_in_lambda_but_reports_original_t(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, family="rectified_flow")
    model_times = np.asarray(config.pilot_model.scales, dtype=np.float64)
    expected_lambda = (1.0 - model_times) / model_times
    observed_coordinates: list[np.ndarray] = []
    observed_preferences: list[str] = []
    original_select = pilot_module.select_stable_scale
    logger = _RecordingLogger()

    def recording_select(coordinates, curves, **kwargs):
        observed_coordinates.append(np.asarray(coordinates).copy())
        observed_preferences.append(kwargs["prefer"])
        return original_select(coordinates, curves, **kwargs)

    monkeypatch.setattr(pilot_module, "select_stable_scale", recording_select)
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
        log_callback=logger,
    )

    assert len(observed_coordinates) == len(PILOT_DATASETS)
    assert observed_preferences == ["smaller"] * len(PILOT_DATASETS)
    for coordinate in observed_coordinates:
        np.testing.assert_allclose(coordinate, expected_lambda)
    for name in PILOT_DATASETS:
        summary = json.loads(
            (run_dir / "datasets" / name / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        selection = summary["scale_selection"]
        index = selection["selected_index"]
        assert selection["selected_t"] == pytest.approx(model_times[index])
        assert selection["selected_scale"] == pytest.approx(model_times[index])
        assert selection["model_scale"] == {
            "name": "t",
            "selected_value": pytest.approx(model_times[index]),
            "prefer": "larger",
        }
        coordinate = selection["selection_coordinate"]
        assert coordinate["name"] == "lambda"
        assert coordinate["formula"] == "(1 - t) / t"
        assert coordinate["prefer"] == "smaller"
        np.testing.assert_allclose(coordinate["values"], expected_lambda)
        assert coordinate["selected_value"] == pytest.approx(
            expected_lambda[index]
        )
        assert selection["model_scale_prefer"] == "larger"
        assert selection["selection_coordinate_prefer"] == "smaller"
        assert selection["prefer"] == "smaller"
    completions = [
        payload
        for event, payload in logger.events
        if event.endswith(".completed") and event != "experiment.completed"
    ]
    assert len(completions) == len(PILOT_DATASETS)
    assert all(
        payload["selection_coordinate"]["name"] == "lambda"
        for payload in completions
    )


@pytest.mark.parametrize(
    ("bad_call", "bad_split"),
    [(1, "validation"), (10, "test")],
)
def test_pilot_refuses_non_finite_prediction_curves_before_metrics_or_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_call: int,
    bad_split: str,
) -> None:
    config = _fixture_config(tmp_path)
    prediction_calls = 0
    metric_calls = 0
    original_metric = pilot_module.known_lid_metrics

    def non_finite_predict(*args, **kwargs):
        nonlocal prediction_calls
        prediction_calls += 1
        prediction = _fake_predict(*args, **kwargs)
        if prediction_calls == bad_call:
            prediction = prediction.copy()
            prediction[0] = np.nan
        return prediction

    def recording_metric(*args, **kwargs):
        nonlocal metric_calls
        metric_calls += 1
        return original_metric(*args, **kwargs)

    monkeypatch.setattr(pilot_module, "known_lid_metrics", recording_metric)
    with pytest.raises(
        FloatingPointError,
        match=rf"{bad_split} prediction curve contains 1 non-finite",
    ):
        run_pilot(
            config,
            root=tmp_path,
            output_root=tmp_path / "runs",
            train_fn=_fake_train,
            predict_fn=non_finite_predict,
        )

    assert metric_calls == 0
    assert not list((tmp_path / "runs").rglob("manifest.json"))


def test_manifest_detects_raw_prediction_tampering(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
    )
    prediction_path = (
        run_dir / "datasets" / PILOT_DATASETS[0] / "test_prediction.npy"
    )
    prediction = np.load(prediction_path, allow_pickle=False)
    prediction[0] += 10.0
    np.save(prediction_path, prediction, allow_pickle=False)
    assert "output inventory does not match manifest" in validate_pilot_experiment(
        run_dir
    )


def test_pilot_config_rejects_dataset_or_project_drift() -> None:
    base = validate_pilot_config(compose_pilot_config(root=REPOSITORY_ROOT))
    wrong_project = OmegaConf.create({**base, "project": "some-other-project"})
    with pytest.raises(PilotConfigError, match="project must be exactly"):
        validate_pilot_config(wrong_project)

    wrong_experiment = OmegaConf.create(
        {**base, "experiment_name": PROJECT_NAME}
    )
    with pytest.raises(PilotConfigError, match="experiment_name must be exactly"):
        validate_pilot_config(wrong_experiment)

    wrong_logging_project = OmegaConf.create(base)
    wrong_logging_project.logging.project = EXPERIMENT_NAME
    with pytest.raises(PilotConfigError, match="logging.project must be exactly"):
        validate_pilot_config(wrong_logging_project)

    wrong_logging_experiment = OmegaConf.create(base)
    wrong_logging_experiment.logging.experiment_name = PROJECT_NAME
    with pytest.raises(
        PilotConfigError, match="logging.experiment_name must be exactly"
    ):
        validate_pilot_config(wrong_logging_experiment)

    wrong_datasets = OmegaConf.create(base)
    wrong_datasets.data.names = list(reversed(PILOT_DATASETS))
    with pytest.raises(PilotConfigError, match="datasets must be exactly"):
        validate_pilot_config(wrong_datasets)


def test_pilot_config_never_accepts_credentials() -> None:
    base = validate_pilot_config(compose_pilot_config(root=REPOSITORY_ROOT))
    base["logging"]["api_key"] = "must-not-enter-a-config"
    with pytest.raises(PilotConfigError, match="credentials are forbidden"):
        validate_pilot_config(base)


def test_pilot_config_requires_every_training_parameter_in_hydra() -> None:
    base = validate_pilot_config(compose_pilot_config(root=REPOSITORY_ROOT))
    base["pilot_model"]["training"].pop("normalization_epsilon")
    with pytest.raises(
        PilotConfigError,
        match="missing pilot_model.training fields:.*normalization_epsilon",
    ):
        validate_pilot_config(base)
