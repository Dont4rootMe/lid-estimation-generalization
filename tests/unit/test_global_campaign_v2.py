from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from kneed import KneeLocator

from experiments import global_campaign_v2 as campaign
from experiments import global_parallel, v2_canary
from experiments.global_parallel_v2 import (
    H100_PROFILE_V2,
    _v2_api,
    with_h100_v2_profile,
)

ROOT = Path(__file__).resolve().parents[2]


def test_v2_config_is_exact_429_matrix_without_batch_override() -> None:
    config = campaign.validate_global_campaign_config(
        campaign.compose_global_campaign_v2_config()
    )
    plans = campaign.model_plans(config)

    assert tuple(plan.variant_id for plan in plans) == campaign.APPROVED_MODEL_VARIANTS
    assert len(plans) * campaign.EXPECTED_GLOBAL_CELL_COUNT == 429
    assert config["logging"]["backend"] == "none"
    assert campaign.CANARY_CELL_KEYS == (
        "e2/e2_uniform_pca/coefficients",
        "e2/e2_arrows/dataset",
    )
    assert campaign.CANARY_COMPANION_CELL_KEYS == (
        "e3/e3_gaussian_pca/coefficients",
        "e1/e1_sampled_fmnist_step1/dataset",
        "e4/e4_sphere_pca_radius1/coefficients",
        "e2/e2_uniform_pca/dataset",
    )
    assert len(campaign.CANARY_TASKS) == 8
    assert len(campaign.CANARY_REUSE_TASKS) == 24
    assert len(set(campaign.CANARY_REUSE_TASKS)) == 24
    assert (
        tuple(f"{variant}/{key}" for variant, key in campaign.CANARY_REUSE_TASKS)
        == v2_canary.EXPECTED_ALL_CELL_IDS
    )
    assert all(plan.model["training"]["batch_size"] == 256 for plan in plans)
    assert plans[0].model["training"]["gradient_clip_norm"] is None
    assert all(
        plan.model["training"]["gradient_clip_norm"] == 1.0 for plan in plans[1:]
    )

    production = with_h100_v2_profile(config)
    assert production["execution"] == {
        "profile": "h100_8gpu_3lane_cell_dag_v2",
        "strategy": "cell_dag_pool",
        "worker_count": 24,
        "physical_device_count": 8,
        "lanes_per_device": 3,
        "training_batch_size_override": None,
        "evaluation_batch_size_override": 512,
    }


def test_v2_h100_profile_maps_three_logical_lanes_to_each_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = tuple(f"GPU-{index}" for index in range(8))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(tokens))

    with _v2_api():
        layout = global_parallel._worker_device_layout(
            H100_PROFILE_V2, require_cuda=True
        )
        with pytest.raises(
            campaign.GlobalCampaignError, match="cannot disable CUDA checks"
        ):
            global_parallel._worker_device_layout(H100_PROFILE_V2, require_cuda=False)

    assert len(layout) == 24
    assert tuple(row["worker_slot"] for row in layout) == tuple(range(24))
    assert tuple(row["visible_device_token"] for row in layout) == tokens * 3
    assert tuple(row["physical_device_index"] for row in layout) == tuple(range(8)) * 3
    assert tuple(row["device_lane"] for row in layout) == (0,) * 8 + (1,) * 8 + (2,) * 8
    assert all(row["multiplexed"] is True for row in layout)

    prepared = SimpleNamespace(
        campaign_identity="b" * 64,
        config={"execution": dict(H100_PROFILE_V2)},
        cells=tuple(range(39)),
        plans=tuple(range(11)),
    )
    records = [
        {
            "slot": row["worker_slot"],
            "visible_token": row["visible_device_token"],
            "cuda_required": True,
            "physical_device_index": row["physical_device_index"],
            "device_lane": row["device_lane"],
            "device_count": 1,
            "device_name": "NVIDIA H100 80GB HBM3",
        }
        for row in reversed(layout)
    ]
    payload = global_parallel._preflight_payload(
        prepared, records=records, layout=layout
    )
    assert payload["schema_version"] == 2
    assert payload["execution"]["worker_count"] == 24
    assert payload["physical_device_count"] == 8
    assert payload["lanes_per_device"] == 3
    assert payload["device_mapping_policy"] == (
        "worker_slot_modulo_physical_device_count_v1"
    )
    assert [row["slot"] for row in payload["workers"]] == list(range(24))
    assert payload["physical_devices"] == [
        {
            "physical_device_index": index,
            "visible_device_token": tokens[index],
            "worker_slots": [index, index + 8, index + 16],
        }
        for index in range(8)
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_count", 8),
        ("worker_count", 23),
        ("physical_device_count", 7),
        ("physical_device_count", True),
        ("lanes_per_device", 2),
        ("lanes_per_device", True),
    ],
)
def test_v2_h100_profile_rejects_any_topology_drift(field: str, value: object) -> None:
    config = with_h100_v2_profile(campaign.compose_global_campaign_v2_config())
    invalid = copy.deepcopy(config)
    invalid["execution"][field] = value
    with pytest.raises(campaign.GlobalCampaignError):
        campaign.validate_global_campaign_config(invalid)


@pytest.mark.parametrize("missing", ["physical_device_count", "lanes_per_device"])
def test_v2_h100_profile_requires_explicit_topology_fields(missing: str) -> None:
    config = with_h100_v2_profile(campaign.compose_global_campaign_v2_config())
    del config["execution"][missing]
    with pytest.raises(campaign.GlobalCampaignError, match="execution fields differ"):
        campaign.validate_global_campaign_config(config)


def test_v2_assignment_ledger_seals_and_validates_slot_device_mapping(
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(
        campaign_identity="a" * 64,
        config={"execution": dict(H100_PROFILE_V2)},
        state_dir=str(tmp_path / "state"),
        plans=(SimpleNamespace(variant_id="vp_diffusion"),),
        cells=(
            SimpleNamespace(key="e2/e2_uniform_pca/coefficients"),
            SimpleNamespace(key="e2/e2_arrows/dataset"),
        ),
    )
    row = {
        "model_index": 0,
        "model_variant": "vp_diffusion",
        "cell_index": 0,
        "cell_key": "e2/e2_uniform_pca/coefficients",
        "worker_slot": 17,
        "visible_device": "GPU-1:NVIDIA H100 80GB HBM3",
        "dispatch_sequence": 1,
        "in_flight_after_dispatch": 24,
        "ready_after_dispatch": 10,
        "status": "completed",
        "physical_device_index": 1,
        "device_lane": 2,
        "visible_device_token": "GPU-1",
    }
    with _v2_api():
        global_parallel._persist_assignments(prepared, {(0, 0): row})
        resumed = global_parallel._resume_assignments(prepared, {(0, 0)})
    assert resumed == {(0, 0): row}

    path = tmp_path / "state" / "parallel" / "assignments.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 2,
        "campaign_identity": "a" * 64,
        "strategy": "cell_dag_pool",
        "worker_count": 24,
        "physical_device_count": 8,
        "lanes_per_device": 3,
        "device_mapping_policy": "worker_slot_modulo_physical_device_count_v1",
        "assignments": [row],
    }

    payload["assignments"][0]["physical_device_index"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with (
        _v2_api(),
        pytest.raises(campaign.GlobalCampaignError, match="assignment row differs"),
    ):
        global_parallel._resume_assignments(prepared, {(0, 0)})

    payload["assignments"][0]["physical_device_index"] = 1
    repeated = copy.deepcopy(payload["assignments"][0])
    repeated["cell_index"] = 1
    repeated["cell_key"] = "e2/e2_arrows/dataset"
    payload["assignments"].append(repeated)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with (
        _v2_api(),
        pytest.raises(
            campaign.GlobalCampaignError, match="repeats a dispatch sequence"
        ),
    ):
        global_parallel._resume_assignments(prepared, {(0, 0), (0, 1)})


def test_v2_pre_run_cell_gate_exactly_binds_all_24_canary_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = campaign.CANARY_CELL_KEYS + campaign.CANARY_COMPANION_CELL_KEYS
    prepared = SimpleNamespace(
        plans=tuple(
            SimpleNamespace(variant_id=variant)
            for variant in campaign.CANARY_MODEL_VARIANTS
        ),
        cells=tuple(SimpleNamespace(key=key) for key in keys),
        preflight_inputs={key: {"source_evidence": {"key": key}} for key in keys},
    )
    task_by_directory: dict[Path, tuple[str, str]] = {}
    validated: list[tuple[str, str]] = []

    def fake_directory(
        *, campaign_root: Path, prepared: object, model_index: int, cell_index: int
    ) -> tuple[Path, dict[str, object]]:
        del campaign_root, prepared
        task = (
            campaign.CANARY_MODEL_VARIANTS[model_index],
            keys[cell_index],
        )
        directory = tmp_path / f"cell-{model_index}-{cell_index}"
        task_by_directory[directory] = task
        return directory, {"task": task}

    def fake_validate(
        directory: Path,
        *,
        expected_identity: object,
        expected_source_evidence: object,
    ) -> list[str]:
        del expected_identity, expected_source_evidence
        validated.append(task_by_directory[directory])
        return []

    def fake_load(path: Path) -> dict[str, object]:
        task = task_by_directory[path.parent]
        status = "passed" if task in set(campaign.CANARY_TASKS) else "not_requested"
        return {"quality_diagnostics": {"status": status}}

    monkeypatch.setattr(campaign, "_expected_cell_directory", fake_directory)
    monkeypatch.setattr(campaign, "validate_global_cell", fake_validate)
    monkeypatch.setattr(campaign, "_load_json", fake_load)

    assert campaign._validate_reusable_canary_cells(tmp_path, prepared) == []
    assert tuple(validated) == campaign.CANARY_REUSE_TASKS


def test_v2_compose_reuses_approved_active_hydra_without_clearing_state() -> None:
    global_hydra = GlobalHydra.instance()
    assert not GlobalHydra.instance().is_initialized()
    with initialize_config_dir(
        version_base="1.3", config_dir=str((ROOT / "configs").resolve())
    ):
        assert global_hydra.is_initialized()
        config = campaign.compose_global_campaign_v2_config()
        assert config.campaign.campaign_id == campaign.APPROVED_CAMPAIGN_ID
        assert global_hydra.is_initialized()
    assert not GlobalHydra.instance().is_initialized()


def test_decorated_canary_reuses_active_hydra_for_nested_v2_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run(config: object, *, config_path: Path) -> None:
        active = GlobalHydra.instance()
        observed["active_before_nested"] = active.is_initialized()
        assert isinstance(config, dict)
        observed["outer_protocol"] = config["protocol_id"]
        observed["search_path"] = tuple(
            (str(entry.provider), str(entry.path))
            for entry in active.hydra.config_loader.get_search_path().config_search_path
        )
        nested = campaign.compose_global_campaign_v2_config()
        observed["campaign_id"] = nested.campaign.campaign_id
        observed["active_after_nested"] = GlobalHydra.instance().is_initialized()
        observed["same_instance"] = GlobalHydra.instance() is active
        observed["config_path"] = config_path

    monkeypatch.setattr(v2_canary, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["experiments.v2_canary"])
    monkeypatch.setenv("LID_V2_CANARY_OUTPUT_ROOT", str(tmp_path / "campaign"))
    monkeypatch.setenv("LID_V2_ARROWS_REVIEWED_GATE", str(tmp_path / "review.json"))
    monkeypatch.setenv("HYDRA_MAIN_MODULE", "__main__")
    assert not GlobalHydra.instance().is_initialized()
    v2_canary._hydra_main()
    assert observed == {
        "active_before_nested": True,
        "outer_protocol": v2_canary.PROTOCOL_ID,
        "search_path": (
            ("hydra", "pkg://hydra.conf"),
            ("main", str((ROOT / "configs").resolve())),
            ("schema", "structured://"),
        ),
        "campaign_id": campaign.APPROVED_CAMPAIGN_ID,
        "active_after_nested": True,
        "same_instance": True,
        "config_path": ROOT / "configs" / "v2_canary.yaml",
    }
    assert not GlobalHydra.instance().is_initialized()


def test_v2_compose_rejects_shadow_active_hydra_without_clearing_state(
    tmp_path: Path,
) -> None:
    shadow = tmp_path / "configs"
    shadow.mkdir()
    (shadow / "global_campaign_v2.yaml").write_text("schema_version: 999\n")
    global_hydra = GlobalHydra.instance()
    assert not GlobalHydra.instance().is_initialized()
    with initialize_config_dir(version_base="1.3", config_dir=str(shadow.resolve())):
        assert global_hydra.is_initialized()
        with pytest.raises(
            campaign.GlobalCampaignError, match="active Hydra search path differs"
        ):
            campaign.compose_global_campaign_v2_config()
        assert global_hydra.is_initialized()
    assert not GlobalHydra.instance().is_initialized()


def test_v2_compose_rejects_unapproved_config_source_without_clearing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialize_config_dir(
        version_base="1.3", config_dir=str((ROOT / "configs").resolve())
    ):
        active = GlobalHydra.instance()
        repository = active.hydra.config_loader.repository
        original_load = repository.load_config

        def shadow_load(config_name: str) -> object:
            if config_name == "global_campaign_v2.yaml":
                return SimpleNamespace(provider="shadow", path="file:///shadow")
            return original_load(config_name)

        monkeypatch.setattr(repository, "load_config", shadow_load)
        with pytest.raises(
            campaign.GlobalCampaignError, match="selected an unapproved source"
        ):
            campaign.compose_global_campaign_v2_config()
        assert GlobalHydra.instance() is active
        assert active.is_initialized()
    assert not GlobalHydra.instance().is_initialized()


def test_vp_common_lambda_roundtrip_and_flipd_filter() -> None:
    values = np.geomspace(campaign.COMMON_LAMBDA_MIN, campaign.COMMON_LAMBDA_MAX, 91)
    times = campaign.vp_time_from_lambda(values)
    np.testing.assert_allclose(campaign.vp_lambda_from_time(times), values, rtol=2e-13)

    reference_lambdas = campaign.unknown_reference_lambdas()
    assert reference_lambdas[0] == campaign.COMMON_LAMBDA_MIN
    assert reference_lambdas[-1] == campaign.COMMON_LAMBDA_MAX
    assert np.all(np.diff(reference_lambdas) > 0.0)
    endpoint_times = campaign.vp_time_from_lambda(
        (campaign.COMMON_LAMBDA_MIN, campaign.COMMON_LAMBDA_MAX)
    )
    np.testing.assert_allclose(
        campaign.vp_time_from_lambda(reference_lambdas),
        np.linspace(endpoint_times[0], endpoint_times[1], 50),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )

    expected = np.linspace(0.0, 1.0, 50)
    expected = expected[(expected > 0.05) & (expected < 0.5)]
    np.testing.assert_array_equal(campaign.flipd_timesteps(), expected)


def test_common_lambda_canonicalization_is_roundoff_only() -> None:
    values = np.asarray(
        [
            np.nextafter(campaign.COMMON_LAMBDA_MIN, 0.0),
            1.0,
            np.nextafter(campaign.COMMON_LAMBDA_MAX, np.inf),
        ]
    )
    canonical = campaign._canonicalize_common_lambdas(values)
    assert canonical[0] == campaign.COMMON_LAMBDA_MIN
    assert canonical[-1] == campaign.COMMON_LAMBDA_MAX
    outside = [campaign.COMMON_LAMBDA_MIN, campaign.COMMON_LAMBDA_MAX]
    for _ in range(5):
        outside[0] = np.nextafter(outside[0], 0.0)
        outside[1] = np.nextafter(outside[1], np.inf)
    for value in outside:
        with pytest.raises(
            campaign.GlobalCampaignError, match="outside declared support"
        ):
            campaign._canonicalize_common_lambdas([value])


@pytest.mark.parametrize(
    "values",
    [
        [campaign.COMMON_LAMBDA_MIN, 1.0, 1.0],
        [campaign.COMMON_LAMBDA_MIN, 2.0, 1.0],
        [np.nextafter(campaign.COMMON_LAMBDA_MIN, 0.0), 1.0],
        [1.0, np.nextafter(campaign.COMMON_LAMBDA_MAX, np.inf)],
    ],
)
def test_saved_common_lambda_grid_requires_canonical_strict_order(
    values: list[float],
) -> None:
    with pytest.raises(
        campaign.GlobalCampaignError, match="canonical and strictly increasing"
    ):
        campaign._require_canonical_common_lambda_grid(values)


def test_flipd_kneedle_matches_pinned_upstream_call_and_fallback() -> None:
    times = campaign.flipd_timesteps()
    curve = np.vstack((np.exp(-8.0 * times), np.ones_like(times)))
    prediction, selected_time, fallback = campaign.flipd_kneedle_pointwise(curve, 30)

    direct = KneeLocator(times, curve[0], S=1.0, curve="convex", direction="decreasing")
    assert direct.knee is not None
    assert prediction[0] == direct.knee_y
    assert selected_time[0] == direct.knee
    assert not fallback[0]
    assert prediction[1] == 30.0
    assert np.isnan(selected_time[1])
    assert fallback[1]


def test_supervised_selector_extends_once_and_ties_to_lower_lambda() -> None:
    lambdas = np.asarray([1.0, 2.0, 4.0])
    curve = np.asarray([[0.2, 0.4, 0.8], [0.2, 0.4, 0.8]])

    extended, values, index, record = campaign.select_supervised_bounded(
        lambdas,
        curve,
        np.zeros(2),
        evaluate=lambda candidate: np.tile(np.where(candidate < 0.4, 0.3, 0.1), (2, 1)),
    )

    assert record["extended_side"] == "lower"
    assert extended[index] == pytest.approx(0.5)
    assert values.shape == (2, len(extended))
    assert record["status"] == "selected"


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_supervised_selector_canonicalizes_support_boundary(side: str) -> None:
    lambdas = campaign.initial_supervised_lambdas()
    if side == "lower":
        curve = np.arange(lambdas.size, dtype=np.float64)[None, :]
    else:
        curve = np.arange(lambdas.size - 1, -1, -1, dtype=np.float64)[None, :]
    seen: list[np.ndarray] = []

    def evaluate(candidates: np.ndarray) -> np.ndarray:
        seen.append(candidates.copy())
        return np.zeros((1, candidates.size), dtype=np.float64)

    extended, _, _, record = campaign.select_supervised_bounded(
        lambdas,
        curve,
        np.zeros(1),
        evaluate=evaluate,
    )

    assert record["extended_side"] == side
    assert len(seen) == 1
    assert seen[0][0] >= campaign.COMMON_LAMBDA_MIN
    assert seen[0][-1] <= campaign.COMMON_LAMBDA_MAX
    boundary = (
        campaign.COMMON_LAMBDA_MIN if side == "lower" else campaign.COMMON_LAMBDA_MAX
    )
    assert boundary in extended


def test_reference_kneedle_seals_explicit_no_knee_failure() -> None:
    scales = campaign.unknown_reference_lambdas()
    index, record = campaign.select_unknown_reference_kneedle(
        scales, np.ones((4, len(scales)), dtype=np.float64)
    )
    assert index is None
    assert record["status"] == "selection_failed"
    assert record["failure_reason"] == "no_knee"
    assert record["selected_lambda"] is None


@pytest.mark.parametrize(
    ("ambient_dim", "expected_width"),
    [(30, 544), (256, 480), (784, 384), (1024, 384), (3072, 448)],
)
def test_nf_width_is_cell_specific_and_within_ten_percent(
    ambient_dim: int, expected_width: int
) -> None:
    width, target, actual, relative_gap = campaign.select_nf_width(ambient_dim)
    assert width == expected_width
    assert actual - target == pytest.approx(relative_gap * target)
    assert abs(relative_gap) <= 0.10


def test_nf_ols5_support_includes_full_stencil() -> None:
    config = campaign.validate_global_campaign_config(
        campaign.compose_global_campaign_v2_config()
    )
    nf = next(
        plan.model
        for plan in campaign.model_plans(config)
        if plan.variant_id == "scale_conditioned_nf"
    )
    campaign._assert_lambdas_supported(
        nf,
        [campaign.COMMON_LAMBDA_MIN, 1.0, campaign.COMMON_LAMBDA_MAX],
        include_nf_stencil=True,
    )
    assert nf["training"]["epsilon_min"] == pytest.approx(
        campaign.COMMON_LAMBDA_MIN * np.exp(-0.1)
    )
    assert nf["training"]["epsilon_max"] == pytest.approx(
        campaign.COMMON_LAMBDA_MAX * np.exp(0.1)
    )
    with pytest.raises(
        campaign.GlobalCampaignError, match="outside declared lambda support"
    ):
        campaign._assert_lambdas_supported(
            nf,
            [campaign.COMMON_LAMBDA_MIN * np.exp(-0.2)],
            include_nf_stencil=True,
        )


def test_unknown_reference_grid_is_native_support_safe_for_all_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = campaign.validate_global_campaign_config(
        campaign.compose_global_campaign_v2_config()
    )
    lambdas = campaign.unknown_reference_lambdas()
    query = np.zeros((3, 30), dtype=np.float64)
    active_model: dict[str, object] = {}
    seen: list[float] = []

    def assert_native_support(model: dict[str, object], scale: float) -> None:
        training = model["training"]
        assert isinstance(training, dict)
        family = model["family"]
        if family == "vp_diffusion":
            assert np.isfinite(campaign.vp_time_from_lambda([scale])).all()
            lower = campaign.COMMON_LAMBDA_MIN
            upper = campaign.COMMON_LAMBDA_MAX
        elif family == "rectified_flow":
            lower = float(training["time_min"])
            upper = float(training["time_max"])
        elif family == "independent_affine_flow":
            lower = float(training["flow_noise_ratio_min"])
            upper = float(training["flow_noise_ratio_max"])
        elif family == "schrodinger_bridge":
            lower = float(training["bridge_tau_min"])
            upper = float(training["bridge_tau_max"])
        else:
            lower = campaign.COMMON_LAMBDA_MIN
            upper = campaign.COMMON_LAMBDA_MAX
        assert lower <= scale <= upper

    def fake_predict(
        _trained: object,
        values: np.ndarray,
        scale: float,
        **_kwargs: object,
    ) -> np.ndarray:
        assert_native_support(active_model, scale)
        seen.append(scale)
        return np.ones(len(values), dtype=np.float64)

    def fake_nf_predict(
        _trained: object,
        values: np.ndarray,
        scale: float,
        **_kwargs: object,
    ) -> np.ndarray:
        assert campaign.COMMON_LAMBDA_MIN <= scale <= campaign.COMMON_LAMBDA_MAX
        seen.append(scale)
        return np.ones(len(values), dtype=np.float64)

    monkeypatch.setattr("models.training.predict_nf_lid_ols5", fake_nf_predict)
    for plan in campaign.model_plans(config):
        model = campaign.resolve_cell_model(plan, {"feature_shape": [30]})
        active_model.clear()
        active_model.update(model)
        seen.clear()
        curve = campaign._prediction_curve(
            fake_predict,
            object(),
            query,
            lambdas,
            model=model,
            seed=0,
            batch_size=128,
            readout=str(model["primary_readout"]),
        )
        assert curve.shape == (3, 50)
        assert np.isfinite(curve).all()
        assert len(seen) == 50


def test_high_dimensional_fm_diagnostics_use_only_shared_hutchinson_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = campaign.validate_global_campaign_config(
        campaign.compose_global_campaign_v2_config()
    )
    plan = next(
        item
        for item in campaign.model_plans(config)
        if item.variant_id == "posterior_log_noise_affine_flow"
    )
    model = campaign.resolve_cell_model(plan, {"feature_shape": [3072]})
    assert model["diagnostics_execution"]["mode"] == ("high_dim_hutchinson_convergence")
    selection_features = np.arange(12 * 3072, dtype=np.float64).reshape(12, 3072)
    partition = campaign.HoldoutPartition(
        fit_indices=np.arange(12, 24, dtype=np.int64),
        selection_indices=np.arange(12, dtype=np.int64),
        fit_features=selection_features.copy(),
        selection_features=selection_features,
        selection_target=np.ones(12, dtype=np.float64),
        record={
            "n_train_selection": 12,
            "selection_features_fm_sha256": campaign.v1._array_sha256(
                selection_features
            ),
        },
    )
    calls: list[tuple[float, int, int]] = []

    def fake_predict(
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
        del trained, family, readout, batch_size
        assert divergence_backend == "hutchinson"
        assert trace_seed == 0
        calls.append((float(scale), int(trace_probes), len(query)))
        return np.full(len(query), float(scale) + trace_probes / 100.0)

    monkeypatch.setattr("models.training.predict_lid", fake_predict)
    scales = campaign.initial_supervised_lambdas()
    curve = np.zeros((12, scales.size), dtype=np.float64)
    output = tmp_path / "fm_diagnostics"
    record = campaign.run_known_affine_diagnostics(
        output,
        trained=SimpleNamespace(checkpoint_sha256="a" * 64),
        partition=partition,
        scales=scales,
        selection_curve=curve,
        model=model,
    )

    assert record["protocol_id"] == campaign.FM_HIGH_DIM_PROTOCOL
    assert record["exact_trace_status"] == "skipped_cost_prohibitive"
    assert record["oracle_status"] == "skipped_cost_prohibitive"
    assert len(calls) == 2 * scales.size
    assert {probes for _scale, probes, _n in calls} == {16, 64}
    assert {size for _scale, _probes, size in calls} == {8}
    assert (
        campaign._validate_high_dim_fm_diagnostics(
            output,
            model=model,
            checkpoint_sha256="a" * 64,
            scales=scales,
            selection_curve=curve,
            partition=partition.record,
            fm=record,
            expected_query_subset_sha256=campaign.v1._array_sha256(
                selection_features[campaign._high_dim_subset_indices(12)]
            ),
        )
        == []
    )
    calls.clear()
    reused = campaign.run_known_affine_diagnostics(
        output,
        trained=SimpleNamespace(checkpoint_sha256="a" * 64),
        partition=partition,
        scales=scales,
        selection_curve=curve,
        model=model,
    )
    assert reused == record
    assert calls == []


@pytest.mark.parametrize(
    ("ambient_dim", "mode"),
    [
        (30, "small_dim_exhaustive_exact"),
        (256, "high_dim_hutchinson_convergence"),
        (784, "high_dim_hutchinson_convergence"),
        (1024, "high_dim_hutchinson_convergence"),
        (3072, "high_dim_hutchinson_convergence"),
    ],
)
def test_fm_diagnostic_cost_policy_is_cell_identity_bearing(
    ambient_dim: int, mode: str
) -> None:
    config = campaign.validate_global_campaign_config(
        campaign.compose_global_campaign_v2_config()
    )
    plan = next(
        item
        for item in campaign.model_plans(config)
        if item.variant_id == "posterior_log_noise_affine_flow"
    )
    model = campaign.resolve_cell_model(plan, {"feature_shape": [ambient_dim]})

    assert model["diagnostics_execution"] == {
        "schema_version": 1,
        "policy": "dimension_aware_known_lid_v2",
        "ambient_dim": ambient_dim,
        "mode": mode,
        "exact_trace_max_ambient_dim": 64,
        "high_dim_query_subset_size": 8,
        "high_dim_trace_probes": [16, 64],
        "high_dim_scale_policy": "actual_selection_grid",
        "high_dim_exact_status": "skipped_cost_prohibitive",
        "high_dim_oracle_status": "skipped_cost_prohibitive",
    }


def test_v2_module_context_does_not_leak_to_v1() -> None:
    key = global_parallel._CAMPAIGN_MODULE_ENV
    before = os.environ.get(key)
    from experiments.global_parallel_v2 import _v2_api

    with _v2_api():
        assert global_parallel._campaign_api() is campaign
    assert os.environ.get(key) == before
