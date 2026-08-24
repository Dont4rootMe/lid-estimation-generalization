from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

import experiments.pilot as pilot_module
from experiments.pilot import (
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
EXPERIMENT_NAMES = {
    "diffusion": (
        "lid-generalization-e8-suite-diffusion-train-mae-scale-selection-seed-137"
    ),
    "rectified_flow": (
        "lid-generalization-e8-suite-rectified-flow-matching-train-mae-"
        "time-selection-seed-137"
    ),
    "scale_conditioned_nf": (
        "lid-generalization-e8-suite-scale-conditioned-normalizing-flow-"
        "train-mae-scale-selection-seed-137"
    ),
    "schrodinger_bridge": (
        "lid-generalization-e8-suite-brownian-schrodinger-bridge-train-mae-"
        "time-selection-seed-137"
    ),
}
CANONICAL_FAMILY = {
    "diffusion": "gaussian_diffusion",
    "rectified_flow": "rectified_flow",
    "scale_conditioned_nf": "scale_conditioned_normalizing_flow",
    "schrodinger_bridge": "brownian_schrodinger_bridge",
}


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
        effective_lid = 1 if name == "e8_spaghetti_pca" else dataset_index
        row: dict[str, Any] = {
            "name": name,
            "representation": "dataset",
            "available_representations": ["dataset"],
            "required_artifacts": ["dataset", "lid"],
            "expected_samples": {"train": 12, "val": 6, "test": 6},
            "expected_shapes": {"dataset": [1, 2, 2]},
            "expected_lid": effective_lid,
            "official": False,
        }
        if name == "e8_spaghetti_pca":
            row["stored_lid_by_split"] = {"train": 20, "val": 20, "test": 1}
            row["lid_override"] = 1
        rows.append(row)
        for split, n_samples in (("train", 12), ("val", 6), ("test", 6)):
            split_dir = root / name / split
            split_dir.mkdir(parents=True)
            # Feature mean identifies the fixture to the fake model without ever
            # passing a LID target through the trainer or selector interface.
            features = np.full(
                (n_samples, 1, 2, 2), float(effective_lid), dtype=np.float32
            )
            features[:, 0, 0, 0] += np.linspace(0.0, 0.01, n_samples)
            np.save(split_dir / "dataset.npy", features, allow_pickle=False)
            stored_lid = (
                {"train": 20, "val": 20, "test": 1}[split]
                if name == "e8_spaghetti_pca"
                else effective_lid
            )
            np.save(
                split_dir / "lid.npy",
                np.full(n_samples, stored_lid, dtype=np.float32),
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
    config = compose_pilot_config(
        [
            f"pilot_model={family}",
            f"data.root={data_root}",
            f"data.registry={registry}",
            "logging.backend=none",
            "pilot_model.training.device=cpu",
            "evaluation.selection.subset_size=4",
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
    assert train.shape[0] == 8
    assert validation.shape[0] == 4
    value = float(np.round(train.mean()))
    checkpoint_path.write_bytes(f"{family}:{value}".encode())
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if log_callback is not None:
        log_callback({"family": family, "epoch": 1, "validation_loss": 0.25})
    return _FakeTrainingResult(
        model={"value": value},
        family=CANONICAL_FAMILY[family],
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


@pytest.mark.parametrize("family", tuple(EXPERIMENT_NAMES))
def test_pilot_trains_three_models_and_seals_all_metrics(
    tmp_path: Path, family: str
) -> None:
    config = _fixture_config(tmp_path, family=family)
    experiment_name = EXPERIMENT_NAMES[family]
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
        payload["step"] == payload["epoch"] == 1 for _, payload in training_events
    )
    assert all(payload["project"] == PROJECT_NAME for _, payload in logger.events)
    assert all(
        payload["experiment_name"] == experiment_name for _, payload in logger.events
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
    assert summary["experiment_name"] == experiment_name
    assert tuple(summary["datasets"]) == PILOT_DATASETS
    assert summary["selection_protocol"] == "held_out_source_train_supervised_v1"
    assert summary["selection_target_split"] == "train_selection"
    assert summary["selection_uses_lid_targets"] is True
    assert summary["selection_uses_validation_targets"] is False
    assert summary["selection_uses_test_targets"] is False
    assert summary["evaluation_protocol"] == "single_train_selected_scale_v1"
    assert summary["retrospective_evaluation_curves_saved"] is False
    assert set(summary["macro_train_selection"]) >= {"mean_mae", "mean_rmse"}
    assert set(summary["macro_validation"]) >= {"mean_mae", "mean_rmse"}
    assert set(summary["macro_test"]) >= {"mean_mae", "mean_rmse"}

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_name"] == experiment_name
    assert manifest["evaluation_protocol"] == "single_train_selected_scale_v1"
    assert manifest["retrospective_evaluation_curves_saved"] is False
    resolved = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved["experiment_name"] == experiment_name
    assert resolved["pilot_model"]["experiment_name"] == experiment_name
    assert resolved["logging"]["experiment_name"] == experiment_name
    spaghetti_overrides = manifest["inputs"]["datasets"]["e8_spaghetti_pca"][
        "applied_overrides"
    ]
    assert spaghetti_overrides == {
        split: {
            "lid": {
                "kind": "constant_after_stored_value_validation",
                "stored": float(stored_lid),
                "effective": 1.0,
            }
        }
        for split, stored_lid in {"train": 20, "val": 20, "test": 1}.items()
    }
    for split, stored_lid in {"train": 20, "val": 20, "test": 1}.items():
        raw_lid = np.load(
            Path(config.data.root) / "e8_spaghetti_pca" / split / "lid.npy",
            allow_pickle=False,
        )
        np.testing.assert_array_equal(raw_lid, np.full(raw_lid.shape, stored_lid))

    registry = yaml.safe_load(
        (run_dir / "artifact_registry.yaml").read_text(encoding="utf-8")
    )
    assert set(registry["artifacts"]) == {f"{name}/dataset" for name in PILOT_DATASETS}
    learned_model = {
        "name": config.pilot_model.name,
        "family": CANONICAL_FAMILY[family],
        "seed": int(config.seed),
        "artifact_registry": str(run_dir / "artifact_registry.yaml"),
        "artifact_registry_sha256": sha256_file(run_dir / "artifact_registry.yaml"),
    }
    loaded_registry = load_artifact_registry(root=tmp_path, model=learned_model)
    for name in PILOT_DATASETS:
        cell = run_dir / "datasets" / name
        for filename in (
            "checkpoint.pt",
            "training.yaml",
            "training_history.json",
            "scales.npy",
            "train_fit_indices.npy",
            "train_selection_indices.npy",
            "train_selection_curve.npy",
            "train_selection_prediction.npy",
            "train_selection_target.npy",
            "validation_prediction.npy",
            "test_prediction.npy",
            "validation_target.npy",
            "test_target.npy",
            "summary.json",
        ):
            assert (cell / filename).is_file()
        assert not (cell / "validation_curve.npy").exists()
        assert not (cell / "test_curve.npy").exists()
        cell_summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
        assert cell_summary["experiment_name"] == experiment_name
        training_config = yaml.safe_load(
            (cell / "training.yaml").read_text(encoding="utf-8")
        )
        assert training_config["experiment_name"] == experiment_name
        assert cell_summary["selection_uses_lid_targets"] is True
        assert cell_summary["selection_uses_validation_targets"] is False
        assert cell_summary["selection_uses_test_targets"] is False
        assert cell_summary["evaluation_protocol"] == "single_train_selected_scale_v1"
        assert cell_summary["frozen_evaluation"] == {
            "schema_version": 1,
            "selected_index": cell_summary["scale_selection"]["selected_index"],
            "selected_scale": cell_summary["scale_selection"]["selected_scale"],
            "validation_candidate_count": 1,
            "test_candidate_count": 1,
            "retrospective_curves_saved": False,
        }
        assert cell_summary["scale_selection"]["uses_ground_truth"] is True
        assert (
            cell_summary["scale_selection"]["ground_truth_split"] == "train_selection"
        )
        assert cell_summary["scale_selection"]["uses_validation_ground_truth"] is False
        assert cell_summary["scale_selection"]["uses_test_ground_truth"] is False
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


def test_train_selection_freezes_before_validation_or_test_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    frozen: set[str] = set()
    prediction_calls: list[tuple[int, float]] = []
    original_load_inputs = pilot_module._load_inputs

    class GuardedSplit:
        def __init__(self, dataset: str, split: str, wrapped: Any) -> None:
            self.dataset = dataset
            self.split = split
            self.wrapped = wrapped

        @property
        def features(self):
            return self.wrapped.features

        @property
        def lid(self):
            if self.split in {"val", "test"}:
                assert self.dataset in frozen, (
                    f"{self.dataset}/{self.split} target was accessed before "
                    "selection.frozen"
                )
            return self.wrapped.lid

    def guarded_load_inputs(**kwargs):
        registry, registry_path, benchmark_root, loaded, record = original_load_inputs(
            **kwargs
        )
        guarded = {
            dataset: {
                split: GuardedSplit(dataset, split, value)
                for split, value in splits.items()
            }
            for dataset, splits in loaded.items()
        }
        return registry, registry_path, benchmark_root, guarded, record

    def logger(event: str, payload: Mapping[str, Any]) -> None:
        if event.endswith(".selection.frozen"):
            frozen.add(str(payload["dataset"]))

    def guarded_predict(*args, **kwargs):
        query = np.asarray(args[1] if len(args) > 1 else kwargs["query"])
        scale = float(args[2] if len(args) > 2 else kwargs["scale"])
        prediction_calls.append((int(query.shape[0]), scale))
        if query.shape[0] == 6:
            assert frozen, "validation/test inference ran before frozen selection"
        return _fake_predict(*args, **kwargs)

    monkeypatch.setattr(pilot_module, "_load_inputs", guarded_load_inputs)
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=guarded_predict,
        log_callback=logger,
    )
    assert frozen == set(PILOT_DATASETS)
    candidate_scales = np.asarray(config.pilot_model.scales, dtype=np.float64)
    calls_per_dataset = candidate_scales.size + 2
    assert len(prediction_calls) == len(PILOT_DATASETS) * calls_per_dataset
    for dataset_index, name in enumerate(PILOT_DATASETS):
        start = dataset_index * calls_per_dataset
        calls = prediction_calls[start : start + calls_per_dataset]
        assert [rows for rows, _ in calls] == [4] * candidate_scales.size + [6, 6]
        np.testing.assert_allclose(
            [scale for _, scale in calls[: candidate_scales.size]], candidate_scales
        )
        summary = json.loads(
            (run_dir / "datasets" / name / "summary.json").read_text("utf-8")
        )
        frozen_scale = float(summary["scale_selection"]["selected_scale"])
        assert [scale for _, scale in calls[-2:]] == pytest.approx(
            [frozen_scale, frozen_scale]
        )


def test_rectified_flow_selects_in_lambda_but_reports_original_t(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path, family="rectified_flow")
    model_times = np.asarray(config.pilot_model.scales, dtype=np.float64)
    expected_lambda = (1.0 - model_times) / model_times
    logger = _RecordingLogger()
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
        log_callback=logger,
    )

    for name in PILOT_DATASETS:
        summary = json.loads(
            (run_dir / "datasets" / name / "summary.json").read_text(encoding="utf-8")
        )
        selection = summary["scale_selection"]
        index = selection["selected_index"]
        assert selection["selected_t"] == pytest.approx(model_times[index])
        assert selection["selected_delta_t"] == pytest.approx(1.0 - model_times[index])
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
        assert coordinate["selected_value"] == pytest.approx(expected_lambda[index])
        assert selection["model_scale_prefer"] == "larger"
        assert selection["selection_coordinate_prefer"] == "smaller"
        assert selection["prefer"] == "larger"
        assert selection["tie_break"] == "larger"
        assert selection["criterion"] == "mae"
        assert selection["ground_truth_split"] == "train_selection"
    completions = [
        payload
        for event, payload in logger.events
        if event.endswith(".completed") and event != "experiment.completed"
    ]
    assert len(completions) == len(PILOT_DATASETS)
    assert all(
        payload["selection_coordinate"]["name"] == "lambda" for payload in completions
    )


def test_normalizing_flow_reports_epsilon_and_log_epsilon(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, family="scale_conditioned_nf")
    scales = np.asarray(config.pilot_model.scales, dtype=np.float64)
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
    )
    for name in PILOT_DATASETS:
        summary = json.loads(
            (run_dir / "datasets" / name / "summary.json").read_text("utf-8")
        )
        selection = summary["scale_selection"]
        index = selection["selected_index"]
        assert selection["model_scale"]["name"] == "epsilon"
        assert selection["selected_epsilon"] == pytest.approx(scales[index])
        assert selection["selected_log_epsilon"] == pytest.approx(np.log(scales[index]))
        assert selection["selection_coordinate"]["name"] == "log_epsilon"


def test_schrodinger_bridge_reports_tau_t_and_brownian_sigma(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path, family="schrodinger_bridge")
    scales = np.asarray(config.pilot_model.scales, dtype=np.float64)
    terminal_time = float(config.pilot_model.training.bridge_terminal_time)
    diffusivity = float(config.pilot_model.training.bridge_diffusivity)
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
    )
    for name in PILOT_DATASETS:
        summary = json.loads(
            (run_dir / "datasets" / name / "summary.json").read_text("utf-8")
        )
        selection = summary["scale_selection"]
        index = selection["selected_index"]
        selected_tau = scales[index]
        assert selection["model_scale"]["name"] == "tau"
        assert selection["selected_tau"] == pytest.approx(selected_tau)
        assert selection["selected_t"] == pytest.approx(terminal_time - selected_tau)
        assert selection["selected_sigma"] == pytest.approx(
            np.sqrt(diffusivity * selected_tau)
        )
        coordinate = selection["selection_coordinate"]
        assert coordinate["name"] == "tau"
        assert coordinate["formula"] == "T - t"
        np.testing.assert_allclose(coordinate["values"], scales)
        assert coordinate["selected_value"] == pytest.approx(selected_tau)


@pytest.mark.parametrize(
    ("bad_call", "bad_split", "error_label", "expected_metric_calls"),
    [
        (1, "train-selection", "train-selection prediction curve", 0),
        (10, "validation", "validation prediction", 10),
        (11, "test", "test prediction", 11),
    ],
)
def test_pilot_refuses_non_finite_predictions_before_metrics_or_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_call: int,
    bad_split: str,
    error_label: str,
    expected_metric_calls: int,
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
        match=rf"{error_label} contains 1 non-finite",
    ):
        run_pilot(
            config,
            root=tmp_path,
            output_root=tmp_path / "runs",
            train_fn=_fake_train,
            predict_fn=non_finite_predict,
        )

    assert bad_split in {"train-selection", "validation", "test"}
    assert metric_calls == expected_metric_calls
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
    prediction_path = run_dir / "datasets" / PILOT_DATASETS[0] / "test_prediction.npy"
    prediction = np.load(prediction_path, allow_pickle=False)
    prediction[0] += 10.0
    np.save(prediction_path, prediction, allow_pickle=False)
    assert "output inventory does not match manifest" in validate_pilot_experiment(
        run_dir
    )


def test_semantic_validator_rejects_resealed_retrospective_evaluation_curve(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
    )
    dataset = PILOT_DATASETS[0]
    forbidden = run_dir / "datasets" / dataset / "validation_curve.npy"
    np.save(
        forbidden,
        np.zeros((6, len(config.pilot_model.scales)), dtype=np.float64),
        allow_pickle=False,
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"] = pilot_module._portable_output_inventory(run_dir)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = validate_pilot_experiment(run_dir)
    assert any(
        "retrospective evaluation artifact is forbidden" in error for error in errors
    )


def test_strict_validation_reconstructs_train_partition_from_source(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
    )
    assert (
        validate_pilot_experiment(
            run_dir,
            verify_inputs=True,
            root=tmp_path,
        )
        == []
    )
    for name in PILOT_DATASETS:
        cell = run_dir / "datasets" / name
        fit = np.load(cell / "train_fit_indices.npy", allow_pickle=False)
        selected = np.load(cell / "train_selection_indices.npy", allow_pickle=False)
        assert np.intersect1d(fit, selected).size == 0
        np.testing.assert_array_equal(
            np.sort(np.concatenate((fit, selected))),
            np.arange(12),
        )


def test_semantic_validator_rejects_resealed_wrong_selected_index(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    run_dir = run_pilot(
        config,
        root=tmp_path,
        output_root=tmp_path / "runs",
        train_fn=_fake_train,
        predict_fn=_fake_predict,
    )
    dataset = PILOT_DATASETS[0]
    cell_summary_path = run_dir / "datasets" / dataset / "summary.json"
    cell_summary = json.loads(cell_summary_path.read_text(encoding="utf-8"))
    n_scales = len(config.pilot_model.scales)
    cell_summary["scale_selection"]["selected_index"] = (
        int(cell_summary["scale_selection"]["selected_index"]) + 1
    ) % n_scales
    cell_summary_path.write_text(
        json.dumps(cell_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate_path = run_dir / "summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["datasets"][dataset] = cell_summary
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"] = pilot_module._portable_output_inventory(run_dir)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = validate_pilot_experiment(run_dir)
    assert any(
        "scale-selection diagnostics do not recompute" in error for error in errors
    )


@pytest.mark.parametrize(("family", "experiment_name"), EXPERIMENT_NAMES.items())
def test_hydra_model_group_is_the_experiment_name_source(
    family: str, experiment_name: str
) -> None:
    config = validate_pilot_config(
        compose_pilot_config([f"pilot_model={family}"], root=REPOSITORY_ROOT)
    )
    assert config["pilot_model"]["experiment_name"] == experiment_name
    assert config["experiment_name"] == experiment_name
    assert config["logging"]["experiment_name"] == experiment_name


@pytest.mark.parametrize("family", tuple(EXPERIMENT_NAMES))
def test_experiment_name_tracks_seed_override(family: str) -> None:
    config = validate_pilot_config(
        compose_pilot_config([f"pilot_model={family}", "seed=23"], root=REPOSITORY_ROOT)
    )
    assert config["experiment_name"].endswith("-seed-23")
    assert config["pilot_model"]["experiment_name"] == config["experiment_name"]


@pytest.mark.parametrize(
    ("family", "backend", "probes"),
    [
        ("diffusion", "hutchinson", 16),
        ("rectified_flow", "hutchinson", 16),
        ("scale_conditioned_nf", "exact", 0),
        ("schrodinger_bridge", "hutchinson", 16),
    ],
)
def test_derivative_contract_is_family_scoped_in_hydra(
    family: str, backend: str, probes: int
) -> None:
    config = validate_pilot_config(
        compose_pilot_config([f"pilot_model={family}"], root=REPOSITORY_ROOT)
    )
    assert config["pilot_model"]["derivative_backend"] == backend
    assert config["pilot_model"]["trace_probes"] == probes
    assert config["evaluation"]["divergence_backend"] == backend
    assert config["evaluation"]["trace_probes"] == probes


def test_scale_conditioned_nf_hydra_schema_is_exact_and_bounded() -> None:
    base = validate_pilot_config(
        compose_pilot_config(["pilot_model=scale_conditioned_nf"], root=REPOSITORY_ROOT)
    )
    training = base["pilot_model"]["training"]
    assert "depth" not in training
    assert "sigma_min" not in training
    assert "time_min" not in training
    assert {
        "num_coupling_layers",
        "conditioner_depth",
        "log_scale_limit",
        "epsilon_min",
        "epsilon_max",
    } <= set(training)

    missing = OmegaConf.create(base)
    missing.pilot_model.training.pop("conditioner_depth")
    with pytest.raises(PilotConfigError, match="conditioner_depth"):
        validate_pilot_config(missing)

    unused = OmegaConf.create(base)
    unused.pilot_model.training.depth = 8
    with pytest.raises(PilotConfigError, match="unknown pilot_model.training"):
        validate_pilot_config(unused)

    invalid_bounds = OmegaConf.create(base)
    invalid_bounds.pilot_model.training.epsilon_min = 2.0
    with pytest.raises(PilotConfigError, match="epsilon_min < epsilon_max"):
        validate_pilot_config(invalid_bounds)

    stochastic_density = OmegaConf.create(base)
    stochastic_density.pilot_model.training.dropout = 0.1
    with pytest.raises(PilotConfigError, match="dropout.*exactly 0"):
        validate_pilot_config(stochastic_density)

    extrapolating_grid = OmegaConf.create(base)
    extrapolating_grid.pilot_model.scales[0] = 0.001
    with pytest.raises(PilotConfigError, match="inside training epsilon bounds"):
        validate_pilot_config(extrapolating_grid)


def test_pilot_config_rejects_dataset_or_project_drift() -> None:
    base = validate_pilot_config(compose_pilot_config(root=REPOSITORY_ROOT))
    wrong_project = OmegaConf.create({**base, "project": "some-other-project"})
    with pytest.raises(PilotConfigError, match="project must be exactly"):
        validate_pilot_config(wrong_project)

    wrong_experiment = OmegaConf.create({**base, "experiment_name": PROJECT_NAME})
    with pytest.raises(
        PilotConfigError,
        match="experiment_name must equal pilot_model.experiment_name",
    ):
        validate_pilot_config(wrong_experiment)

    wrong_logging_project = OmegaConf.create(base)
    wrong_logging_project.logging.project = base["experiment_name"]
    with pytest.raises(PilotConfigError, match="logging.project must be exactly"):
        validate_pilot_config(wrong_logging_project)

    wrong_logging_experiment = OmegaConf.create(base)
    wrong_logging_experiment.logging.experiment_name = PROJECT_NAME
    with pytest.raises(PilotConfigError, match="logging.experiment_name must equal"):
        validate_pilot_config(wrong_logging_experiment)

    wrong_model_experiment = OmegaConf.create(base)
    wrong_model_experiment.pilot_model.experiment_name = "normalizing-flow"
    with pytest.raises(
        PilotConfigError, match="pilot_model.experiment_name must be exactly"
    ):
        validate_pilot_config(wrong_model_experiment)

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
