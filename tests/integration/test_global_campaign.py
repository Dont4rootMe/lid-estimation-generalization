from __future__ import annotations

import json
import subprocess
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


def test_imported_hydra_entrypoint_can_resolve_every_nested_pilot_model(
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


def test_python_module_entrypoint_resolves_nested_models_before_source_preflight(
    tmp_path: Path,
) -> None:
    missing_archive = tmp_path / "missing-benchmarks.zip"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.global_campaign",
            "logging.backend=none",
            f"data.canonical_archive={missing_archive}",
            f"output_root={tmp_path / 'output'}",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    combined_output = completed.stdout + completed.stderr
    assert "cannot compose approved pilot model" not in combined_output
    assert (
        "canonical archive/extracted tree failed exact verification" in combined_output
    )


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


def test_all_model_training_configs_accept_materialized_checkpoint_defaults() -> None:
    config = validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    plans = global_campaign.model_plans(config)

    assert tuple(plan.variant_id for plan in plans) == (
        global_campaign.APPROVED_MODEL_VARIANTS
    )
    for plan in plans:
        materialized = global_campaign._canonical_training_config_record(
            plan.model["training"], field="test Hydra training config"
        )
        assert len(materialized) > len(plan.model["training"])
        global_campaign._require_matching_training_configs(
            materialized, plan.model["training"]
        )


def test_training_config_guard_rejects_real_or_unknown_checkpoint_changes() -> None:
    config = validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    plan = global_campaign.model_plans(config)[0]
    materialized = global_campaign._canonical_training_config_record(
        plan.model["training"], field="test Hydra training config"
    )

    changed = dict(materialized)
    changed["learning_rate"] = float(changed["learning_rate"]) * 2.0
    with pytest.raises(
        GlobalCampaignError,
        match="trained checkpoint config differs from the Hydra model config",
    ):
        global_campaign._require_matching_training_configs(
            changed, plan.model["training"]
        )

    unknown = dict(materialized)
    unknown["undeclared_setting"] = None
    with pytest.raises(
        GlobalCampaignError,
        match="trained checkpoint config is not a valid TrainingConfig",
    ):
        global_campaign._require_matching_training_configs(
            unknown, plan.model["training"]
        )


def _tiny_plans(config: dict[str, Any]) -> tuple[ModelPlan, ...]:
    model = {
        "name": "test_diffusion",
        "family": "diffusion",
        "readout": "full",
        "selection_prefer": "smaller",
        "derivative_backend": "exact",
        "trace_probes": 0,
        "training": {
            "device": "cpu",
            "seed": 0,
            "epochs": 1,
            "early_stopping_patience": None,
        },
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
        del family, validation
        self.calls += 1
        progress = Path(progress_checkpoint_path)
        self.progress_paths.append(progress)
        if self.fail_call == self.calls:
            progress.write_bytes(b"strict-progress-fixture")
            self.fail_call = None
            raise RuntimeError("fixture interruption")
        checkpoint_record = {
            "call": self.calls,
            "train_sum": float(np.asarray(train).sum()),
            "config": global_campaign._canonical_training_config_record(
                config, field="fake trainer config"
            ),
        }
        Path(checkpoint_path).write_text(
            json.dumps(checkpoint_record, sort_keys=True), encoding="utf-8"
        )
        progress.unlink(missing_ok=True)
        return SimpleNamespace(
            checkpoint_path=Path(checkpoint_path),
            checkpoint_sha256=sha256_path(Path(checkpoint_path)),
            config=global_campaign._canonical_training_config_record(
                config, field="fake trainer config"
            ),
            history=(
                {
                    "epoch": 1,
                    "train_loss": 0.5,
                    "validation_loss": 0.25,
                    "learning_rate": 1.0e-3,
                },
            ),
            best_epoch=1,
            best_validation_loss=0.25,
        )


def _load_checkpoint(path, *, device):
    del device
    checkpoint_record = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimpleNamespace(
        checkpoint_path=Path(path),
        checkpoint_sha256=sha256_path(Path(path)),
        config=checkpoint_record["config"],
        history=(
            {
                "epoch": 1,
                "train_loss": 0.5,
                "validation_loss": 0.25,
                "learning_rate": 1.0e-3,
            },
        ),
        best_epoch=1,
        best_validation_loss=0.25,
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


def test_pruning_rejects_missing_or_nonminimal_training_history(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"training-attestation-fixture")
    checkpoint_sha = sha256_path(checkpoint)
    model = _tiny_plans(
        {
            "campaign": {
                "models": [{"id": "diffusion", "experiment_name": "fixture-diffusion"}]
            }
        }
    )[0].model

    historyless = SimpleNamespace(history=None)
    with pytest.raises(GlobalCampaignError, match="complete training history"):
        global_campaign._training_attestation(
            historyless,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            retention_policy="prune_after_cell_evaluation",
            model=model,
        )

    nonminimal = SimpleNamespace(
        history=(
            {
                "epoch": 1,
                "train_loss": 0.5,
                "validation_loss": 1.0,
                "learning_rate": 1.0e-3,
            },
            {
                "epoch": 2,
                "train_loss": 0.4,
                "validation_loss": 9.0,
                "learning_rate": 1.0e-3,
            },
        ),
        best_epoch=2,
        best_validation_loss=9.0,
    )
    nonminimal_model = {
        **model,
        "training": {
            **model["training"],
            "epochs": 2,
            "early_stopping_patience": None,
        },
    }
    with pytest.raises(GlobalCampaignError, match="first strict minimum"):
        global_campaign._training_attestation(
            nonminimal,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            retention_policy="prune_after_cell_evaluation",
            model=nonminimal_model,
        )

    valid = SimpleNamespace(
        history=nonminimal.history,
        best_epoch=1,
        best_validation_loss=1.0,
    )
    attestation = global_campaign._training_attestation(
        valid,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        retention_policy="prune_after_cell_evaluation",
        model=nonminimal_model,
    )
    attestation["history"]["best_epoch"] = 2
    attestation["history"]["best_validation_loss"] = 9.0
    errors = global_campaign._validate_training_attestation(
        attestation,
        model=nonminimal_model,
        checkpoint_retention="prune_after_cell_evaluation",
        checkpoint_sha256=checkpoint_sha,
    )
    assert any("first strict minimum" in error for error in errors)


def test_pruned_affine_cell_rejects_resealed_cross_checkpoint_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import fm_diagnostics

    monkeypatch.setattr(fm_diagnostics, "validate_fm_diagnostics", lambda path: [])
    config = validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    cell = load_campaign_inventory(config, PROJECT_ROOT)[13]
    assert cell.target_policy == "known_lid"
    data = _cell_loader_factory({"value": 1})(cell, config, PROJECT_ROOT)
    model = global_campaign._resolved_pilot_model("direct_rectified_flow", 0)
    model["training"] = {
        **model["training"],
        "device": "cpu",
        "epochs": 1,
        "early_stopping_patience": None,
    }
    plan = ModelPlan(
        variant_id="direct_rectified_flow",
        experiment_name="affine-cross-binding-fixture",
        model=model,
    )
    trainer = _FakeTraining()

    def fake_diagnostics(
        output_dir,
        *,
        trained,
        partition,
        scales,
        selection_curve,
        model,
    ):
        output = Path(output_dir)
        arrays = output / "arrays"
        arrays.mkdir(parents=True)
        response = np.zeros_like(selection_curve)
        correction = np.asarray(selection_curve)
        for name, value in {
            "scales": scales,
            "target": partition.selection_target,
            "full": selection_curve,
            "response": response,
            "correction": correction,
        }.items():
            global_campaign._save_npy(arrays / f"{name}.npy", value)
        metadata = {
            "checkpoint_sha256": trained.checkpoint_sha256,
            "outer_selection_curve_sha256": global_campaign._array_sha256(
                selection_curve
            ),
            "raw_query_sha256": global_campaign._array_sha256(
                partition.selection_features
            ),
            "variant_id": model["training"]["flow_variant_id"],
            "config": model["diagnostics"],
        }
        global_campaign._write_json(output / "metadata.json", metadata)
        global_campaign._write_json(output / "summary.json", {"fixture": True})
        global_campaign._write_json(output / "manifest.json", {"fixture": True})
        return {
            "status": "completed_strict_v2",
            "path": "fm_diagnostics",
            "manifest_sha256": sha256_path(output / "manifest.json"),
            "metadata_sha256": sha256_path(output / "metadata.json"),
            "summary_sha256": sha256_path(output / "summary.json"),
            "outer_selection_curve_sha256": global_campaign._array_sha256(
                selection_curve
            ),
        }

    final_dir, _ = global_campaign._run_cell(
        campaign_root=tmp_path / "campaign",
        campaign_id=str(config["campaign"]["campaign_id"]),
        campaign_config_sha="1" * 64,
        source_sha=SOURCE_SHA,
        config=config,
        model_plan=plan,
        cell=cell,
        data=data,
        reference_summary=None,
        train_fn=trainer,
        predict_fn=_predict,
        load_checkpoint_fn=_load_checkpoint,
        affine_diagnostics_fn=fake_diagnostics,
        callback=None,
    )
    assert global_campaign.validate_global_cell(final_dir) == []

    diagnostic_dir = final_dir / "fm_diagnostics"
    metadata = global_campaign._load_json(diagnostic_dir / "metadata.json")
    metadata["checkpoint_sha256"] = "f" * 64
    global_campaign._write_json(diagnostic_dir / "metadata.json", metadata)
    summary = global_campaign._load_json(final_dir / "summary.json")
    summary["fm_diagnostics"]["metadata_sha256"] = sha256_path(
        diagnostic_dir / "metadata.json"
    )
    summary["fm_diagnostics"]["manifest_sha256"] = sha256_path(
        diagnostic_dir / "manifest.json"
    )
    global_campaign._write_json(final_dir / "summary.json", summary)
    manifest = global_campaign._load_json(final_dir / "manifest.json")
    manifest["outputs"] = global_campaign._output_inventory(
        final_dir, excluded_relative_paths=frozenset({"checkpoint.pt"})
    )
    global_campaign._write_json(final_dir / "manifest.json", manifest)

    errors = global_campaign.validate_global_cell(final_dir)
    assert "FM diagnostics are not bound to the outer checkpoint" in errors

    metadata["checkpoint_sha256"] = summary["checkpoint_sha256"]
    metadata["outer_selection_curve_sha256"] = "e" * 64
    global_campaign._write_json(diagnostic_dir / "metadata.json", metadata)
    summary["fm_diagnostics"]["metadata_sha256"] = sha256_path(
        diagnostic_dir / "metadata.json"
    )
    summary["fm_diagnostics"]["outer_selection_curve_sha256"] = "e" * 64
    global_campaign._write_json(final_dir / "summary.json", summary)
    manifest["outputs"] = global_campaign._output_inventory(
        final_dir, excluded_relative_paths=frozenset({"checkpoint.pt"})
    )
    global_campaign._write_json(final_dir / "manifest.json", manifest)

    errors = global_campaign.validate_global_cell(final_dir)
    assert "FM diagnostics are not bound to the outer selection curve" in errors


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
    for record in manifest["cells"]:
        cell_dir = campaign_root / record["path"]
        assert not (cell_dir / "checkpoint.pt").exists()
        attestation = json.loads((cell_dir / "training_attestation.json").read_text())
        summary = json.loads((cell_dir / "summary.json").read_text())
        assert attestation["checkpoint_retention"] == ("prune_after_cell_evaluation")
        assert attestation["checkpoint_sha256"] == summary["checkpoint_sha256"]
        assert attestation["history"]["status"] == "complete"
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
    attestation_path = known_dir / "training_attestation.json"
    original_attestation = global_campaign._load_json(attestation_path)
    tampered_attestation = {
        **original_attestation,
        "checkpoint_sha256": "0" * 64,
    }
    global_campaign._write_json(attestation_path, tampered_attestation)
    cell_manifest = global_campaign._load_json(known_dir / "manifest.json")
    cell_manifest["outputs"] = global_campaign._output_inventory(known_dir)
    global_campaign._write_json(known_dir / "manifest.json", cell_manifest)
    attestation_errors = global_campaign.validate_global_cell(
        known_dir, expected_identity=cell_manifest["identity"]
    )
    assert any(
        "attestation checkpoint SHA differs" in error for error in attestation_errors
    )
    global_campaign._write_json(attestation_path, original_attestation)
    cell_manifest["outputs"] = global_campaign._output_inventory(known_dir)
    global_campaign._write_json(known_dir / "manifest.json", cell_manifest)

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


def test_legacy_retained_checkpoint_contract_remains_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    config["campaign"]["evaluation"].pop("checkpoint_retention")
    config = validate_global_campaign_config(config)
    monkeypatch.setattr(global_campaign, "model_plans", _tiny_plans)
    monkeypatch.setattr(
        global_campaign, "hash_declared_sources", lambda root: SOURCE_SHA
    )
    trainer = _FakeTraining(fail_call=2)
    revision = {"value": 1}
    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_global_campaign(
            config,
            root=PROJECT_ROOT,
            output_root=tmp_path,
            source_preflight_fn=_source_preflight,
            cell_loader=_cell_loader_factory(revision),
            train_fn=trainer,
            predict_fn=_predict,
            load_checkpoint_fn=_load_checkpoint,
            logger_factory=_logger_factory([], []),
        )
    campaign_root = next(path for path in tmp_path.iterdir() if path.is_dir())
    sealed = [
        path.parent
        for path in campaign_root.rglob("manifest.json")
        if not path.parent.name.startswith(".")
    ]
    assert len(sealed) == 1
    assert (sealed[0] / "checkpoint.pt").is_file()
    assert global_campaign.validate_global_cell(sealed[0]) == []
    # Pre-pruning schema-v1 cells had neither the optional attestation nor the
    # explicit summary retention fields; they remain strictly valid.
    (sealed[0] / "training_attestation.json").unlink()
    summary = global_campaign._load_json(sealed[0] / "summary.json")
    summary.pop("checkpoint_retention")
    summary.pop("training_attestation_sha256")
    global_campaign._write_json(sealed[0] / "summary.json", summary)
    manifest = global_campaign._load_json(sealed[0] / "manifest.json")
    manifest["outputs"] = global_campaign._output_inventory(sealed[0])
    global_campaign._write_json(sealed[0] / "manifest.json", manifest)
    assert global_campaign.validate_global_cell(sealed[0]) == []


def test_pruned_complete_staging_recovers_without_retraining(
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
    trainer = _FakeTraining(fail_call=2)
    replace = global_campaign.os.replace
    interrupt = {"armed": True}

    def interrupt_first_cell_seal(source, destination):
        source_path = Path(source)
        if (
            interrupt["armed"]
            and source_path.is_dir()
            and source_path.name.startswith(".")
            and source_path.name.endswith(".incomplete")
        ):
            interrupt["armed"] = False
            assert not (source_path / "checkpoint.pt").exists()
            raise RuntimeError("fixture post-prune seal interruption")
        return replace(source, destination)

    monkeypatch.setattr(global_campaign.os, "replace", interrupt_first_cell_seal)
    kwargs = {
        "root": PROJECT_ROOT,
        "output_root": tmp_path,
        "source_preflight_fn": _source_preflight,
        "cell_loader": _cell_loader_factory(revision),
        "train_fn": trainer,
        "predict_fn": _predict,
        "load_checkpoint_fn": _load_checkpoint,
        "logger_factory": _logger_factory([], []),
    }
    with pytest.raises(RuntimeError, match="post-prune seal interruption"):
        run_global_campaign(config, **kwargs)
    campaign_root = next(path for path in tmp_path.iterdir() if path.is_dir())
    staging = next(campaign_root.rglob(".*.incomplete"))
    assert (staging / "manifest.json").is_file()
    assert not (staging / "checkpoint.pt").exists()
    assert global_campaign.validate_global_cell(staging) == []

    # Cell 1 is sealed directly from its validated staging directory.  The
    # trainer's second call belongs to cell 2 and proves cell 1 was not rerun.
    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_global_campaign(config, **kwargs)
    assert trainer.calls == 2
    ledger = json.loads((campaign_root / "state" / "ledger.json").read_text())
    assert len(ledger["completed_cells"]) == 1
    sealed = campaign_root / ledger["completed_cells"][0]["path"]
    assert not (sealed / "checkpoint.pt").exists()
    assert global_campaign.validate_global_cell(sealed) == []


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
        (
            "campaign.evaluation.checkpoint_retention=delete_immediately",
            "checkpoint_retention",
        ),
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
