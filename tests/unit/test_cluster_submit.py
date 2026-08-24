from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys
import types

import pytest

from experiments.cluster_submit import (
    APPROVED_FAMILIES,
    CUBLAS_WORKSPACE_CONFIG,
    ClusterConfig,
    ClusterConfigError,
    ClusterSubmissionError,
    INSTANCE_TYPE,
    JOB_DESC,
    build_job_payload,
    load_cluster_config,
    main,
    plan_jobs,
    submit_jobs,
    validate_job_payload,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "cluster" / "shared_a100.yaml"


def _config() -> ClusterConfig:
    return load_cluster_config(CONFIG)


def test_planned_jobs_pin_all_scheduler_fair_use_metadata() -> None:
    jobs = plan_jobs(_config())
    assert tuple(job.family for job in jobs) == APPROVED_FAMILIES
    assert len(jobs) == 2
    for job in jobs:
        payload = job.payload
        assert payload["job_desc"] == JOB_DESC
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
        assert payload["env_variables"]["COMET_PROJECT_NAME"] == (
            "lid-generalization"
        )
        assert payload["env_variables"]["COMET_EXPERIMENT_NAME"] == (
            "ent-block-diffusion-eval"
        )
        metadata_text = repr(
            {key: value for key, value in payload.items() if key != "script"}
        )
        # ``diffusion`` occurs only inside the mandated project namespace and
        # the pre-approved base-image identifier; neither encodes this run's
        # selected family.
        metadata_text = metadata_text.replace(JOB_DESC, "")
        metadata_text = metadata_text.replace(
            str(payload["base_image"]), ""
        ).replace("ent-block-diffusion-eval", "").replace(
            "lid-generalization", ""
        )
        assert planned.family not in metadata_text
        assert f"pilot_model={planned.family}" in payload["script"]
        assert "logging.project=lid-generalization" in payload["script"]
        assert (
            "logging.experiment_name=ent-block-diffusion-eval"
            in payload["script"]
        )
        assert "conda activate block-diff" in payload["script"]
        assert "COMET_API_KEY=$(" in payload["script"]
        assert " output_root=" in payload["script"]
        assert " output_dir=" not in payload["script"]


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


def test_model_or_dataset_identity_in_env_metadata_is_rejected() -> None:
    payload = build_job_payload(_config(), "diffusion")
    payload["env_variables"]["RUN_KIND"] = "rectified_flow"
    with pytest.raises(ClusterConfigError, match="model/dataset identity"):
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
    assert output.count(JOB_DESC) == 2
    assert "must-not-be-here" not in output


def test_submit_constructs_exactly_two_jobs_after_mode_check(
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
    assert len(job_names) == len(submitted) == len(constructed) == 2
    assert job_names == ("lm-mpi-job-safe-1", "lm-mpi-job-safe-2")
    assert {payload["job_desc"] for payload in constructed} == {JOB_DESC}
    assert {payload["queue_name"] for payload in constructed} == {"shared"}
    assert {payload["priority_class"] for payload in constructed} == {
        "shared-medium"
    }


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

    monkeypatch.setitem(
        sys.modules, "client_lib", types.SimpleNamespace(Job=FailedJob)
    )
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
        lambda config: ("lm-mpi-job-one", "lm-mpi-job-two"),
    )
    assert main(["--config", str(CONFIG), "--submit"]) == 0
    output = capsys.readouterr().out
    assert "status: submitted" in output
    assert "project: lid-generalization" in output
    assert "experiment_name: ent-block-diffusion-eval" in output
    assert "lm-mpi-job-one" in output
    assert "lm-mpi-job-two" in output
    assert "COMET_API_KEY" not in output
