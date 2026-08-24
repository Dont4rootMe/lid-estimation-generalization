from __future__ import annotations

import importlib
import re
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.cluster_submit import (
    CUBLAS_WORKSPACE_CONFIG,
    INSTANCE_TYPE,
    JOB_DESC,
    PILOT_ENTRYPOINT,
    ClusterConfig,
    ClusterConfigError,
    ClusterSubmissionError,
    build_job_payload,
    load_cluster_config,
    main,
    plan_jobs,
    submit_jobs,
    validate_job_payload,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "cluster" / "shared_a100.yaml"
FM_CONFIG = ROOT / "configs" / "cluster" / "shared_a100_fm.yaml"
EXPERIMENT_NAMES = {
    "diffusion": (
        "lid-generalization-e8-suite-diffusion-train-mae-scale-selection-seed-0"
    ),
    "rectified_flow": (
        "lid-generalization-e8-suite-rectified-flow-matching-"
        "train-mae-time-selection-seed-0"
    ),
    "scale_conditioned_nf": (
        "lid-generalization-e8-suite-scale-conditioned-normalizing-flow-"
        "train-mae-scale-selection-seed-0"
    ),
    "schrodinger_bridge": (
        "lid-generalization-e8-suite-brownian-schrodinger-bridge-"
        "train-mae-time-selection-seed-0"
    ),
}
FM_EXPERIMENT_NAMES = {
    "direct_rectified_flow": (
        "lid-generalization-e8-suite-fm-rectified-direct-velocity-all-readouts-"
        "debug-train-mae-lambda-selection-seed-0"
    ),
    "posterior_rectified_flow": (
        "lid-generalization-e8-suite-fm-rectified-posterior-mean-all-readouts-"
        "debug-train-mae-lambda-selection-seed-0"
    ),
    "direct_log_noise_affine_flow": (
        "lid-generalization-e8-suite-fm-log-noise-direct-velocity-all-readouts-"
        "debug-train-mae-lambda-selection-seed-0"
    ),
    "posterior_log_noise_affine_flow": (
        "lid-generalization-e8-suite-fm-log-noise-posterior-mean-all-readouts-"
        "debug-train-mae-lambda-selection-seed-0"
    ),
    "direct_vp_trigonometric_flow": (
        "lid-generalization-e8-suite-fm-vp-trigonometric-direct-velocity-"
        "all-readouts-debug-train-mae-lambda-selection-seed-0"
    ),
    "posterior_vp_trigonometric_flow": (
        "lid-generalization-e8-suite-fm-vp-trigonometric-posterior-mean-"
        "all-readouts-debug-train-mae-lambda-selection-seed-0"
    ),
}


def _config() -> ClusterConfig:
    return load_cluster_config(CONFIG)


def _fm_config() -> ClusterConfig:
    return load_cluster_config(FM_CONFIG)


def test_fm_campaign_plans_exactly_six_declared_factorial_jobs() -> None:
    jobs = plan_jobs(_fm_config())
    assert tuple(job.family for job in jobs) == tuple(FM_EXPERIMENT_NAMES)
    assert {job.family: job.experiment_name for job in jobs} == FM_EXPERIMENT_NAMES
    assert len(jobs) == 6
    for job in jobs:
        payload = job.payload
        assert payload["job_desc"] == JOB_DESC
        assert payload["queue_name"] == "shared"
        assert payload["priority_class"] == "shared-medium"
        assert payload["instance_type"] == "a100.1gpu"
        assert "COMET_API_KEY" not in payload["env_variables"]
        assert f"pilot_model={job.family}" in payload["script"]
        assert f"logging.experiment_name={job.experiment_name}" in payload["script"]


def test_cluster_config_scope_rejects_approved_but_undeclared_family() -> None:
    with pytest.raises(ClusterConfigError, match="not declared"):
        build_job_payload(_config(), "direct_rectified_flow")
    with pytest.raises(ClusterConfigError, match="not declared"):
        plan_jobs(_fm_config(), "diffusion")


def test_planned_jobs_pin_all_scheduler_fair_use_metadata() -> None:
    jobs = plan_jobs(_config())
    assert tuple(job.family for job in jobs) == tuple(_config().execution["families"])
    assert {job.family: job.experiment_name for job in jobs} == EXPERIMENT_NAMES
    assert len(jobs) == 4
    for job in jobs:
        payload = job.payload
        assert payload["job_desc"] == JOB_DESC
        assert payload["job_desc"] == (
            "echimbulatov | ent-block-diffusion-eval #ID0137 #rnd"
        )
        assert payload["queue_name"] == "shared"
        assert payload["priority_class"] == "shared-medium"
        assert payload["region"] == "A100-MT"
        assert payload["type"] == "pytorch2"
        assert payload["preflight_check"] is True
        assert payload["n_workers"] == 1
        assert payload["instance_type"] == INSTANCE_TYPE
        assert payload["flags"] == {}
        assert (
            payload["env_variables"]["CUBLAS_WORKSPACE_CONFIG"]
            == CUBLAS_WORKSPACE_CONFIG
        )


def test_payload_has_no_serialized_secret_or_family_scheduler_metadata() -> None:
    for planned in plan_jobs(_config()):
        payload = planned.payload
        assert "COMET_API_KEY" not in payload["env_variables"]
        assert payload["env_variables"]["COMET_PROJECT_NAME"] == ("lid-generalization")
        assert "COMET_EXPERIMENT_NAME" not in payload["env_variables"]
        assert payload["env_variables"]["COMET_WORKSPACE"] == "dont4rootme"
        assert payload["env_variables"]["COMET_CONFIG"] == (
            "/home/jovyan/.comet.config"
        )
        metadata_text = repr(
            {key: value for key, value in payload.items() if key != "script"}
        )
        # The fixed legacy job description and base-image identifier are not
        # the selected Comet experiment name.
        metadata_text = metadata_text.replace(JOB_DESC, "")
        metadata_text = metadata_text.replace(str(payload["base_image"]), "").replace(
            "lid-generalization", ""
        )
        assert planned.family not in metadata_text
        assert f"pilot_model={planned.family}" in payload["script"]
        assert "logging.project=lid-generalization" in payload["script"]
        assert (
            f"logging.experiment_name={EXPERIMENT_NAMES[planned.family]}"
            in payload["script"]
        )
        assert "logging.workspace=dont4rootme" in payload["script"]
        assert payload["script"].startswith(f"{PILOT_ENTRYPOINT} ")
        assert "\n" not in payload["script"]
        assert not re.search(r"[$;&|`<>()]", payload["script"])
        assert "COMET_" not in payload["script"]
        assert " output_root=" in payload["script"]
        assert " output_dir=" not in payload["script"]


@pytest.mark.parametrize(
    "suffix",
    ["\nwhoami", ";whoami", " && whoami", " $(whoami)", " | whoami"],
)
def test_script_shell_syntax_is_rejected(suffix: str) -> None:
    payload = build_job_payload(_config(), "diffusion")
    payload["script"] += suffix
    with pytest.raises(ClusterConfigError, match="safe command"):
        validate_job_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_desc", JOB_DESC + " "),
        ("queue_name", "shared "),
        ("priority_class", "shared-medium "),
        ("region", "A100-MT "),
        ("type", "pytorch2 "),
        ("preflight_check", False),
        ("n_workers", 2),
        ("instance_type", "a100.8gpu"),
    ],
)
def test_payload_tampering_fails_closed(field: str, value: object) -> None:
    payload = build_job_payload(_config(), "diffusion")
    payload[field] = value
    with pytest.raises(ClusterConfigError):
        validate_job_payload(payload)


@pytest.mark.parametrize(
    "identity",
    [
        "diffusion",
        "rectified_flow",
        "rectified-flow-matching",
        "scale_conditioned_nf",
        "schrodinger_bridge",
    ],
)
def test_model_or_dataset_identity_in_env_metadata_is_rejected(
    identity: str,
) -> None:
    payload = build_job_payload(_config(), "diffusion")
    payload["env_variables"]["RUN_KIND"] = identity
    with pytest.raises(ClusterConfigError, match="model/dataset identity"):
        validate_job_payload(payload)


def test_global_comet_experiment_environment_is_rejected() -> None:
    payload = build_job_payload(_config(), "diffusion")
    payload["env_variables"]["COMET_EXPERIMENT_NAME"] = "diffusion"
    with pytest.raises(ClusterConfigError, match="Hydra owns the name"):
        validate_job_payload(payload)


def test_secret_field_in_scheduler_environment_is_rejected() -> None:
    payload = build_job_payload(_config(), "diffusion")
    payload["env_variables"]["COMET_API_KEY"] = "must-not-be-here"
    with pytest.raises(ClusterConfigError, match="secret variable"):
        validate_job_payload(payload)


@pytest.mark.parametrize("value", [None, "", ":16:8", ":4096:2"])
def test_deterministic_cublas_workspace_config_is_mandatory(
    value: str | None,
) -> None:
    payload = build_job_payload(_config(), "diffusion")
    if value is None:
        del payload["env_variables"]["CUBLAS_WORKSPACE_CONFIG"]
    else:
        payload["env_variables"]["CUBLAS_WORKSPACE_CONFIG"] = value
    with pytest.raises(
        ClusterConfigError,
        match=r"environment\.CUBLAS_WORKSPACE_CONFIG must be ':4096:8'",
    ):
        validate_job_payload(payload)


def test_unapproved_family_is_rejected() -> None:
    with pytest.raises(ClusterConfigError, match="unapproved pilot family"):
        build_job_payload(_config(), "normalizing_flow")


def test_cross_family_experiment_name_is_rejected() -> None:
    payload = build_job_payload(_config(), "diffusion")
    payload["script"] = payload["script"].replace(
        f"logging.experiment_name={EXPERIMENT_NAMES['diffusion']}",
        f"logging.experiment_name={EXPERIMENT_NAMES['rectified_flow']}",
    )
    with pytest.raises(ClusterConfigError, match="Hydra overrides"):
        validate_job_payload(payload)


def test_single_family_plan_is_exactly_one_rectified_flow_job() -> None:
    jobs = plan_jobs(_config(), "rectified_flow")
    assert len(jobs) == 1
    assert jobs[0].family == "rectified_flow"
    assert jobs[0].experiment_name == EXPERIMENT_NAMES["rectified_flow"]
    assert jobs[0].payload["job_desc"] == JOB_DESC


def test_dry_run_is_default_and_does_not_import_client_lib(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = importlib.import_module

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "client_lib":
            raise AssertionError("dry run must not import client_lib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    assert main(["--config", str(CONFIG)]) == 0
    output = capsys.readouterr().out
    assert "mode: dry-run" in output
    assert output.count(JOB_DESC) == 4
    assert "must-not-be-here" not in output


def test_dry_run_can_select_only_rectified_flow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--config", str(CONFIG), "--family", "rectified_flow"]) == 0
    output = capsys.readouterr().out
    assert output.count(JOB_DESC) == 1
    assert "family: rectified_flow" in output
    assert f"experiment_name: {EXPERIMENT_NAMES['rectified_flow']}" in output
    assert "pilot_model=diffusion" not in output


def test_submit_constructs_exactly_four_jobs_after_mode_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_config = tmp_path / ".comet.config"
    private_config.write_text("[comet]\napi_key=not-read-by-launcher\n")
    private_config.chmod(0o600)
    config = _config()
    config = replace(
        config,
        execution={
            **config.execution,
            "comet_config_file": str(private_config),
        },
    )
    constructed: list[dict[str, object]] = []
    submitted: list[object] = []

    class FakeJob:
        counter = 0

        def __init__(self, **payload: object) -> None:
            constructed.append(payload)
            self.job_name: str | None = None

        def submit(self) -> str:
            type(self).counter += 1
            self.job_name = f"lm-mpi-job-safe-{type(self).counter}"
            submitted.append(self)
            return f'Job "{self.job_name}" created.'

    monkeypatch.setitem(sys.modules, "client_lib", types.SimpleNamespace(Job=FakeJob))
    job_names = submit_jobs(config)
    assert len(job_names) == len(submitted) == len(constructed) == 4
    assert job_names == tuple(f"lm-mpi-job-safe-{index}" for index in range(1, 5))
    assert {payload["job_desc"] for payload in constructed} == {JOB_DESC}
    assert {payload["queue_name"] for payload in constructed} == {"shared"}
    assert {payload["priority_class"] for payload in constructed} == {"shared-medium"}


def test_submit_rejects_readable_by_group_secret_file(tmp_path: Path) -> None:
    private_config = tmp_path / ".comet.config"
    private_config.write_text("[comet]\napi_key=not-read-by-launcher\n")
    private_config.chmod(0o640)
    config = _config()
    config = replace(
        config,
        execution={
            **config.execution,
            "comet_config_file": str(private_config),
        },
    )
    with pytest.raises(ClusterConfigError, match="mode 0600"):
        submit_jobs(config)


def test_submit_rejects_client_error_return_without_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_config = tmp_path / ".comet.config"
    private_config.write_text("[comet]\napi_key=not-read-by-launcher\n")
    private_config.chmod(0o600)
    config = replace(
        _config(),
        execution={
            **_config().execution,
            "comet_config_file": str(private_config),
        },
    )

    class FailedJob:
        job_name = None

        def __init__(self, **payload: object) -> None:
            pass

        def submit(self) -> str:
            return "Error 400: rejected"

    monkeypatch.setitem(sys.modules, "client_lib", types.SimpleNamespace(Job=FailedJob))
    with pytest.raises(ClusterSubmissionError) as caught:
        submit_jobs(config)
    assert caught.value.accepted_job_names == ()
    assert "rejected" not in str(caught.value)


def test_submit_reports_already_accepted_name_on_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_config = tmp_path / ".comet.config"
    private_config.write_text("[comet]\napi_key=not-read-by-launcher\n")
    private_config.chmod(0o600)
    base = _config()
    config = replace(
        base,
        execution={
            **base.execution,
            "comet_config_file": str(private_config),
        },
    )

    class PartialJob:
        created = 0

        def __init__(self, **payload: object) -> None:
            type(self).created += 1
            self.ordinal = type(self).created
            self.job_name: str | None = None

        def submit(self) -> str:
            if self.ordinal == 1:
                self.job_name = "lm-mpi-job-accepted-first"
                return f'Job "{self.job_name}" created.'
            return "Error 503: unavailable"

    monkeypatch.setitem(
        sys.modules, "client_lib", types.SimpleNamespace(Job=PartialJob)
    )
    with pytest.raises(ClusterSubmissionError) as caught:
        submit_jobs(config)
    assert caught.value.accepted_job_names == ("lm-mpi-job-accepted-first",)
    assert "1 job(s) were already accepted" in str(caught.value)
    assert "unavailable" not in str(caught.value)


def test_submit_wraps_client_exception_without_leaking_its_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_config = tmp_path / ".comet.config"
    private_config.write_text("[comet]\napi_key=not-read-by-launcher\n")
    private_config.chmod(0o600)
    base = _config()
    config = replace(
        base,
        execution={
            **base.execution,
            "comet_config_file": str(private_config),
        },
    )

    class ExplodingJob:
        def __init__(self, **payload: object) -> None:
            pass

        def submit(self) -> str:
            raise RuntimeError("untrusted-server-message")

    monkeypatch.setitem(
        sys.modules, "client_lib", types.SimpleNamespace(Job=ExplodingJob)
    )
    with pytest.raises(ClusterSubmissionError) as caught:
        submit_jobs(config)
    assert caught.value.accepted_job_names == ()
    assert "untrusted-server-message" not in str(caught.value)


def test_submit_cli_prints_only_safe_acknowledged_job_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "experiments.cluster_submit.submit_jobs",
        lambda config, family=None: tuple(
            f"lm-mpi-job-{index}" for index in range(1, 5)
        ),
    )
    assert main(["--config", str(CONFIG), "--submit"]) == 0
    output = capsys.readouterr().out
    assert "status: submitted" in output
    assert "project: lid-generalization" in output
    for family, experiment_name in EXPERIMENT_NAMES.items():
        assert f"{family}: {experiment_name}" in output
    for index in range(1, 5):
        assert f"lm-mpi-job-{index}" in output
    assert "COMET_API_KEY" not in output


def test_submit_cli_forwards_single_family_selector(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    selected: list[str | None] = []

    def fake_submit(
        config: ClusterConfig, family: str | None = None
    ) -> tuple[str, ...]:
        selected.append(family)
        return ("lm-mpi-job-rf",)

    monkeypatch.setattr("experiments.cluster_submit.submit_jobs", fake_submit)
    assert (
        main(
            [
                "--config",
                str(CONFIG),
                "--family",
                "rectified_flow",
                "--submit",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert selected == ["rectified_flow"]
    assert f"rectified_flow: {EXPERIMENT_NAMES['rectified_flow']}" in output
    assert "diffusion: diffusion" not in output
