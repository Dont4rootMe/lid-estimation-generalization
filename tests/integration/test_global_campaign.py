from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from experiments import global_campaign
from experiments.comet_logging import CometConfigurationError
from experiments.global_campaign import (
    APPROVED_GLOBAL_CELL_KEYS,
    CellData,
    GlobalCampaignError,
    ModelLoggerHandle,
    ModelPlan,
    compose_global_campaign_config,
    load_campaign_inventory,
    run_global_campaign,
    validate_global_campaign,
    validate_global_campaign_config,
)
from experiments.run_manifest import canonical_json, sha256_bytes, sha256_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "4" * 64


def test_real_hydra_entrypoint_can_resolve_every_nested_pilot_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, Any] = {}

    def validate_only(config):
        hydra_before = GlobalHydra.instance().hydra
        observed["config"] = validate_global_campaign_config(config)
        observed["plans"] = global_campaign.model_plans(observed["config"])
        assert GlobalHydra.instance().hydra is hydra_before
        return tmp_path / "validated"

    monkeypatch.setattr(global_campaign, "run_global_campaign", validate_only)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "global_campaign",
            "logging.backend=none",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
    )

    global_campaign.main()

    assert (
        tuple(row["id"] for row in observed["config"]["campaign"]["models"])
        == global_campaign.APPROVED_MODEL_VARIANTS
    )
    assert tuple(plan.variant_id for plan in observed["plans"]) == (
        global_campaign.APPROVED_MODEL_VARIANTS
    )
    assert not GlobalHydra.instance().is_initialized()


def test_nested_pilot_compose_rejects_an_unapproved_active_hydra_path(
    tmp_path: Path,
) -> None:
    with (
        initialize_config_dir(version_base="1.3", config_dir=str(tmp_path.resolve())),
        pytest.raises(
            GlobalCampaignError,
            match="cannot compose approved pilot model",
        ) as error,
    ):
        global_campaign._resolved_pilot_model("diffusion", 0)
    assert isinstance(error.value.__cause__, GlobalCampaignError)
    assert "active Hydra search path differs" in str(error.value.__cause__)


def test_nested_pilot_compose_rejects_an_extra_package_search_provider() -> None:
    config_dir = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        search_path = GlobalHydra.instance().hydra.config_loader.get_search_path()
        search_path.prepend(provider="unapproved", path="pkg://unapproved.configs")
        with pytest.raises(
            GlobalCampaignError,
            match="cannot compose approved pilot model",
        ) as error:
            global_campaign._resolved_pilot_model("diffusion", 0)
    assert isinstance(error.value.__cause__, GlobalCampaignError)
    assert "active Hydra search path differs" in str(error.value.__cause__)


def test_nested_pilot_compose_rejects_a_shadowed_pilot_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        repository = GlobalHydra.instance().hydra.config_loader.repository
        original_load_config = repository.load_config

        def shadow_pilot(config_name: str):
            if config_name == "pilot.yaml":
                return SimpleNamespace(provider="main", path="pkg://experiments")
            return original_load_config(config_name)

        monkeypatch.setattr(repository, "load_config", shadow_pilot)
        with pytest.raises(
            GlobalCampaignError,
            match="cannot compose approved pilot model",
        ) as error:
            global_campaign._resolved_pilot_model("diffusion", 0)
    assert isinstance(error.value.__cause__, GlobalCampaignError)
    assert "selected an unapproved source" in str(error.value.__cause__)


def _tiny_plans(config: dict[str, Any]) -> tuple[ModelPlan, ...]:
    model = {
        "name": "test_diffusion",
        "family": "diffusion",
        "readout": "full",
        "selection_prefer": "smaller",
        "derivative_backend": "exact",
        "trace_probes": 0,
        "training": {"device": "cpu", "seed": 0},
        "scales": [0.1, 0.2, 0.4],
    }
    return tuple(
        ModelPlan(
            variant_id=row["id"],
            experiment_name=row["experiment_name"],
            model=model,
        )
        for row in config["campaign"]["models"]
    )


def _source_preflight(config, root, cells):
    del config, root
    return {
        cell.inventory_id: {
            "schema_version": 1,
            "kind": "integration_test_seal",
            "inventory_id": cell.inventory_id,
        }
        for cell in cells
    }


def _cell_loader_factory(revision: dict[str, int]):
    def load(cell, config, root):
        del config, root
        ordinal = APPROVED_GLOBAL_CELL_KEYS.index(cell.key)
        if cell.target_policy == "paired_delta":
            offset = float(cell.expected_lid_delta)
        elif cell.target_policy == "sample_size":
            offset = float(ordinal) / 100.0
        else:
            offset = float(ordinal) / 1000.0
        train = np.column_stack((np.full(12, offset), np.linspace(0.0, 1.0, 12)))
        validation = np.column_stack((np.full(4, offset), np.arange(4)))
        test = np.column_stack((np.full(4, offset), np.arange(4)))
        known = cell.target_policy == "known_lid"
        train_target = train[:, 0] + 0.2 if known else None
        validation_target = validation[:, 0] + 0.2 if known else None
        test_target = test[:, 0] + 0.2 if known else None
        labels = np.arange(4, dtype=np.int64)
        record = {
            "schema_version": 1,
            "cell_key": cell.key,
            "fixture_revision": revision["value"],
        }
        return CellData(
            train=train,
            validation=validation,
            test=test,
            train_target=train_target,
            validation_target=validation_target,
            test_target=test_target,
            validation_labels=labels,
            test_labels=labels,
            input_record=record,
            input_sha256=sha256_bytes(canonical_json(record).encode()),
        )

    return load


class _FakeTraining:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call
        self.progress_paths: list[Path] = []

    def __call__(
        self,
        family,
        train,
        validation,
        config,
        checkpoint_path,
        log_callback=None,
        *,
        progress_checkpoint_path=None,
    ):
        del family, validation, config
        self.calls += 1
        progress = Path(progress_checkpoint_path)
        self.progress_paths.append(progress)
        if self.fail_call == self.calls:
            progress.write_bytes(b"strict-progress-fixture")
            self.fail_call = None
            raise RuntimeError("fixture interruption")
        Path(checkpoint_path).write_bytes(
            f"checkpoint:{self.calls}:{float(np.asarray(train).sum())}".encode()
        )
        progress.unlink(missing_ok=True)
        return SimpleNamespace(
            checkpoint_path=Path(checkpoint_path),
            checkpoint_sha256=sha256_path(Path(checkpoint_path)),
        )


def _load_checkpoint(path, *, device):
    del device
    return SimpleNamespace(
        checkpoint_path=Path(path), checkpoint_sha256=sha256_path(Path(path))
    )


def _predict(
    trained,
    query,
    scale,
    *,
    family,
    readout,
    divergence_backend,
    trace_probes,
    trace_seed,
    batch_size,
):
    del (
        trained,
        family,
        readout,
        divergence_backend,
        trace_probes,
        trace_seed,
        batch_size,
    )
    values = np.asarray(query)
    return values[:, 0] + float(scale)


def _logger_factory(events, closes):
    def factory(plan, campaign_identity, state_dir, logging):
        del campaign_identity, state_dir, logging

        def callback(event, payload):
            events.append((plan.variant_id, event, payload))

        def close():
            closes.append(plan.variant_id)

        return ModelLoggerHandle(
            callback,
            close,
            None,
            log_asset=lambda path, name: None,
        )

    return factory


def test_all_models_all_cells_resume_and_deep_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    cells = load_campaign_inventory(config, PROJECT_ROOT)
    assert tuple(cell.key for cell in cells) == APPROVED_GLOBAL_CELL_KEYS
    monkeypatch.setattr(global_campaign, "model_plans", _tiny_plans)
    monkeypatch.setattr(
        global_campaign, "hash_declared_sources", lambda root: SOURCE_SHA
    )
    revision = {"value": 1}
    trainer = _FakeTraining(fail_call=6)
    events: list[tuple[str, str, Any]] = []
    closes: list[str] = []
    kwargs = {
        "root": PROJECT_ROOT,
        "output_root": tmp_path,
        "source_preflight_fn": _source_preflight,
        "cell_loader": _cell_loader_factory(revision),
        "train_fn": trainer,
        "predict_fn": _predict,
        "load_checkpoint_fn": _load_checkpoint,
        "logger_factory": _logger_factory(events, closes),
    }
    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_global_campaign(config, **kwargs)
    interrupted_progress = trainer.progress_paths[-1]
    assert interrupted_progress.is_file()

    campaign_root = run_global_campaign(config, **kwargs)
    assert trainer.calls == 391
    assert trainer.progress_paths[5] == trainer.progress_paths[6]
    assert not interrupted_progress.exists()
    assert (
        validate_global_campaign(
            campaign_root,
            project_root=PROJECT_ROOT,
            source_preflight_fn=_source_preflight,
            cell_loader=_cell_loader_factory(revision),
        )
        == []
    )
    manifest = json.loads((campaign_root / "campaign.json").read_text())
    assert manifest["expected_models"] == 10
    assert manifest["expected_cells_per_model"] == 39
    assert len(manifest["cells"]) == 390
    aggregate = json.loads((campaign_root / "aggregate.json").read_text())
    assert aggregate["coverage"]["models"] == 10
    assert aggregate["e1_sample_size_stability"]
    assert aggregate["e5_paired_delta"]
    assert (campaign_root / "unified_results.csv").is_file()
    assert all(
        row["metrics"]["mae"] == pytest.approx(0.0)
        for row in aggregate["e5_paired_delta"]
    )
    assert len({row["model_variant"] for row in aggregate["known_lid"]}) == 10

    # Completed science is reused and only the durable telemetry replay phase runs.
    calls_before = trainer.calls
    assert run_global_campaign(config, **kwargs) == campaign_root
    assert trainer.calls == calls_before

    known_cell = next(cell for cell in cells if cell.target_policy == "known_lid")
    known_record = next(
        row
        for row in manifest["cells"]
        if row["model_variant"] == "diffusion" and row["cell_key"] == known_cell.key
    )
    known_dir = campaign_root / known_record["path"]
    target_path = known_dir / "test_target.npy"
    target = np.load(target_path, allow_pickle=False)
    np.save(target_path, target + 1.0, allow_pickle=False)
    cell_manifest = global_campaign._load_json(known_dir / "manifest.json")
    cell_manifest["outputs"] = global_campaign._output_inventory(known_dir)
    global_campaign._write_json(known_dir / "manifest.json", cell_manifest)
    fresh_data = global_campaign._bind_source_preflight(
        _cell_loader_factory(revision)(known_cell, config, PROJECT_ROOT),
        known_cell,
        _source_preflight(config, PROJECT_ROOT, cells),
    )
    evidence_errors = global_campaign.validate_global_cell(
        known_dir,
        expected_identity=cell_manifest["identity"],
        expected_source_evidence=global_campaign._source_evidence(fresh_data, config),
    )
    assert any(
        "test target differs from freshly loaded source" in error
        for error in evidence_errors
    )
    with pytest.raises(GlobalCampaignError, match="existing final global campaign"):
        run_global_campaign(config, **kwargs)


def test_input_mutation_changes_campaign_identity_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    monkeypatch.setattr(global_campaign, "model_plans", _tiny_plans)
    monkeypatch.setattr(
        global_campaign, "hash_declared_sources", lambda root: SOURCE_SHA
    )
    revision = {"value": 1}

    class StopImmediately(_FakeTraining):
        def __call__(
            self,
            family,
            train,
            validation,
            config,
            checkpoint_path,
            log_callback=None,
            *,
            progress_checkpoint_path=None,
        ):
            del (
                family,
                train,
                validation,
                config,
                checkpoint_path,
                log_callback,
                progress_checkpoint_path,
            )
            raise RuntimeError("stop after identity")

    kwargs = {
        "root": PROJECT_ROOT,
        "output_root": tmp_path,
        "source_preflight_fn": _source_preflight,
        "cell_loader": _cell_loader_factory(revision),
        "train_fn": StopImmediately(),
        "predict_fn": _predict,
        "load_checkpoint_fn": _load_checkpoint,
    }
    with pytest.raises(RuntimeError, match="stop after identity"):
        run_global_campaign(config, **kwargs)
    first = {path.name for path in tmp_path.iterdir() if path.is_dir()}
    revision["value"] = 2
    with pytest.raises(RuntimeError, match="stop after identity"):
        run_global_campaign(config, **kwargs)
    second = {path.name for path in tmp_path.iterdir() if path.is_dir()}
    assert len(first) == 1
    assert len(second) == 2


def test_durable_spool_replays_transient_open_and_event_failures(
    tmp_path: Path,
) -> None:
    plan = ModelPlan(
        variant_id="diffusion", experiment_name="descriptive-diffusion", model={}
    )
    identity = "a" * 64

    def offline(*args, **kwargs):
        raise TimeoutError("network unavailable")

    pending = global_campaign._safe_model_logger(
        offline,
        plan,
        identity,
        tmp_path / "state",
        {"backend": "comet"},
    )
    pending.callback("model.started", {"model": "diffusion"})
    pending.close()
    assert pending.telemetry_record()["events_pending"] == 1

    delivered: list[tuple[str, Any]] = []
    close_calls: list[bool] = []

    def online(*args, **kwargs):
        return ModelLoggerHandle(
            lambda event, payload: delivered.append((event, payload)),
            lambda: close_calls.append(True),
            global_campaign._deterministic_experiment_key(identity, "diffusion"),
        )

    resumed = global_campaign._safe_model_logger(
        online,
        plan,
        identity,
        tmp_path / "state",
        {"backend": "comet"},
    )
    resumed.close()
    assert [event for event, _ in delivered] == ["model.started"]
    assert resumed.telemetry_record()["events_pending"] == 0
    assert close_calls == [True]


def test_non_transient_comet_configuration_error_is_fail_closed(
    tmp_path: Path,
) -> None:
    plan = ModelPlan("diffusion", "descriptive-diffusion", {})

    def invalid(*args, **kwargs):
        raise CometConfigurationError("bad credential file")

    with pytest.raises(GlobalCampaignError, match="configuration/dependency"):
        global_campaign._safe_model_logger(
            invalid,
            plan,
            "b" * 64,
            tmp_path / "state",
            {"backend": "comet"},
        )


def test_comet_state_resumes_same_deterministic_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: dict[str, Any] = {}
    opens: list[tuple[str, str]] = []

    class Experiment:
        def __init__(self, *, experiment_key, project_name, workspace):
            del project_name, workspace
            self.key = experiment_key
            self.names: list[str] = []
            created[experiment_key] = self
            opens.append(("new", experiment_key))

        def get_key(self):
            return self.key

        def set_name(self, name):
            self.names.append(name)

        def add_tag(self, tag):
            pass

        def log_metrics(self, metrics, *, step=None, prefix=None):
            pass

        def log_parameters(self, parameters, *, prefix=None):
            pass

        def log_asset(self, file_data, *, file_name=None):
            pass

        def end(self):
            pass

    class ExistingExperiment(Experiment):
        def __init__(self, *, experiment_key, project_name, workspace):
            del project_name, workspace
            existing = created[experiment_key]
            self.__dict__ = existing.__dict__
            opens.append(("existing", experiment_key))

    fake_module = SimpleNamespace(
        Experiment=Experiment, ExistingExperiment=ExistingExperiment
    )
    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    monkeypatch.setattr(
        "experiments.comet_logging.require_comet_environment", lambda: None
    )
    plan = ModelPlan("diffusion", "descriptive-diffusion", {})
    logging = {"backend": "comet"}
    first = global_campaign.open_model_logger(
        plan, "c" * 64, tmp_path / "state", logging
    )
    first.close()
    second = global_campaign.open_model_logger(
        plan, "c" * 64, tmp_path / "state", logging
    )
    second.close()
    expected_key = global_campaign._deterministic_experiment_key("c" * 64, "diffusion")
    assert first.experiment_key == second.experiment_key == expected_key
    assert opens == [("new", expected_key), ("existing", expected_key)]
    state = global_campaign._load_json(tmp_path / "state" / "comet" / "diffusion.json")
    assert state["registration_status"] == "ready"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("schema_version=true", "schema_version"),
        ("campaign.resume.schema_version=true", "resume"),
        ("campaign.selection.tie_tolerance=false", "tie_tolerance"),
        (
            "campaign.selection.stability_min_valid_fraction=true",
            "stability_min_valid_fraction",
        ),
        ("campaign.evaluation.frozen_candidate_count=true", "one frozen"),
    ],
)
def test_boolean_numeric_contracts_are_rejected(override: str, match: str) -> None:
    with pytest.raises(GlobalCampaignError, match=match):
        validate_global_campaign_config(
            compose_global_campaign_config((override, "logging.backend=none"))
        )


@pytest.mark.parametrize("name", ["foo", "unrelated-name"])
def test_generic_comet_experiment_names_are_rejected(name: str) -> None:
    with pytest.raises(GlobalCampaignError, match="approved name"):
        validate_global_campaign_config(
            compose_global_campaign_config(
                (
                    f"campaign.models.0.experiment_name={name}",
                    "logging.backend=none",
                )
            )
        )


def test_inventory_root_drift_is_rejected_before_source_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    cells = list(load_campaign_inventory(config, PROJECT_ROOT))
    generated_index = next(
        index for index, cell in enumerate(cells) if cell.source_kind != "exact_archive"
    )
    cells[generated_index] = replace(
        cells[generated_index], data_root="data/wrong-generated-root"
    )

    fake_archive = SimpleNamespace(
        archive_size_bytes=1,
        file_count=1,
        uncompressed_size_bytes=1,
    )
    monkeypatch.setattr(
        "datasets.archive.verify_exact_archive", lambda path: fake_archive
    )
    monkeypatch.setattr(
        "datasets.archive.verify_extracted_tree", lambda root, manifest: None
    )
    with pytest.raises(GlobalCampaignError, match="inconsistent source identity"):
        global_campaign.validate_campaign_sources(config, PROJECT_ROOT, cells)
