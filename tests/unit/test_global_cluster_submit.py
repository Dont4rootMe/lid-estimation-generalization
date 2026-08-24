from __future__ import annotations

import importlib
import re
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.cluster_submit import INSTANCE_TYPE, JOB_DESC
from experiments.global_cluster_submit import (
    GLOBAL_CAMPAIGN,
    GLOBAL_ENTRYPOINT,
    GlobalClusterConfigError,
    GlobalClusterSubmissionError,
    build_global_job_payload,
    load_global_cluster_config,
    main,
    plan_global_job,
    submit_global_job,
    validate_global_job_payload,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "cluster" / "shared_a100_global.yaml"


def _config():
    return load_global_cluster_config(CONFIG)


def test_global_campaign_plans_exactly_one_job() -> None:
    planned = plan_global_job(_config())
    payload = planned.payload
    assert planned.campaign == GLOBAL_CAMPAIGN
    assert payload["job_desc"] == JOB_DESC
    assert payload["queue_name"] == "shared"
    assert payload["priority_class"] == "shared-medium"
    assert payload["region"] == "A100-MT"
    assert payload["type"] == "pytorch2"
    assert payload["preflight_check"] is True
    assert payload["n_workers"] == 1
    assert payload["instance_type"] == INSTANCE_TYPE
    assert payload["flags"] == {}
    assert payload["script"].startswith(f"{GLOBAL_ENTRYPOINT} ")
    assert "campaign=all_suites_all_models" in payload["script"]
    assert "logging.project=lid-generalization" in payload["script"]
    assert "logging.workspace=dont4rootme" in payload["script"]
    assert "logging.experiment_name=" not in payload["script"]
    assert "\n" not in payload["script"]
    assert not re.search(r"[$;&|`<>()]", payload["script"])


def test_global_payload_has_no_secret_or_model_identity_in_metadata() -> None:
    payload = dict(plan_global_job(_config()).payload)
    assert "COMET_API_KEY" not in payload["env_variables"]
    assert "COMET_EXPERIMENT_NAME" not in payload["env_variables"]
    assert payload["env_variables"]["COMET_CONFIG"] == "/home/jovyan/.comet.config"
    assert payload["env_variables"]["COMET_PROJECT_NAME"] == "lid-generalization"
    assert payload["env_variables"]["COMET_WORKSPACE"] == "dont4rootme"
    metadata_text = repr(
        {
            key: value
            for key, value in payload.items()
            if key not in {"script", "job_desc", "base_image"}
        }
    ).lower()
    for identity in (
        "diffusion",
        "rectified",
        "flow_matching",
        "schrodinger",
        "normalizing_flow",
        "spaghetti",
        "sphere4",
        "fmnist",
    ):
        assert identity not in metadata_text


@pytest.mark.parametrize(
    "suffix",
    ["\nwhoami", ";whoami", " && whoami", " $(whoami)", " | whoami"],
)
def test_global_script_shell_syntax_is_rejected(suffix: str) -> None:
    payload = build_global_job_payload(_config())
    payload["script"] += suffix
    with pytest.raises(GlobalClusterConfigError, match="safe command"):
        validate_global_job_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_desc", JOB_DESC + " "),
        ("queue_name", "shared "),
        ("base_image", "example.invalid/image:latest"),
        ("priority_class", "shared-medium "),
        ("region", "A100-MT "),
        ("type", "pytorch2 "),
        ("preflight_check", False),
        ("preflight_check", 1),
        ("n_workers", 2),
        ("n_workers", True),
        ("instance_type", "a100.8gpu"),
    ],
)
def test_global_payload_tampering_fails_closed(field: str, value: object) -> None:
    payload = build_global_job_payload(_config())
    payload[field] = value
    with pytest.raises(GlobalClusterConfigError):
        validate_global_job_payload(payload)


@pytest.mark.parametrize(
    "identity",
    ["diffusion", "rectified", "flow_matching", "schrodinger", "fmnist"],
)
def test_model_or_dataset_identity_in_global_env_is_rejected(
    identity: str,
) -> None:
    payload = build_global_job_payload(_config())
    payload["env_variables"]["RUN_KIND"] = identity
    with pytest.raises(GlobalClusterConfigError, match="exact allowlist"):
        validate_global_job_payload(payload)


def test_arbitrary_credential_like_environment_is_rejected() -> None:
    payload = build_global_job_payload(_config())
    payload["env_variables"]["CREDENTIAL"] = "sk-must-not-enter-the-payload"
    with pytest.raises(GlobalClusterConfigError, match="exact allowlist"):
        validate_global_job_payload(payload)


def test_secret_and_global_comet_name_in_environment_are_rejected() -> None:
    payload = build_global_job_payload(_config())
    payload["env_variables"]["COMET_API_KEY"] = "must-not-be-here"
    with pytest.raises(GlobalClusterConfigError, match="secret variable"):
        validate_global_job_payload(payload)

    payload = build_global_job_payload(_config())
    payload["env_variables"]["COMET_EXPERIMENT_NAME"] = "global"
    with pytest.raises(GlobalClusterConfigError, match="Hydra owns names"):
        validate_global_job_payload(payload)


def test_cross_campaign_or_model_override_is_rejected() -> None:
    payload = build_global_job_payload(_config())
    payload["script"] = payload["script"].replace(
        "campaign=all_suites_all_models", "campaign=diffusion"
    )
    with pytest.raises(GlobalClusterConfigError, match="global allowlist"):
        validate_global_job_payload(payload)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("scheduler", "n_workers"), True),
        (("scheduler", "n_gpus"), True),
        (("scheduler", "preflight_check"), 1),
    ],
)
def test_global_cluster_yaml_rejects_bool_integer_aliases(
    tmp_path: Path, path: tuple[str, ...], replacement: object
) -> None:
    import yaml

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cursor = raw
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    candidate = tmp_path / "cluster.yaml"
    candidate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(GlobalClusterConfigError):
        load_global_cluster_config(candidate)


def test_global_dry_run_is_default_and_does_not_import_client_lib(
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
    assert "job_count: 1" in output
    assert output.count(JOB_DESC) == 1
    assert "must-not-be-here" not in output


def test_submit_constructs_and_submits_exactly_one_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    checked_secret_paths: list[str] = []
    monkeypatch.setattr(
        "experiments.global_cluster_submit._validate_runtime_secret_file",
        checked_secret_paths.append,
    )
    constructed: list[dict[str, object]] = []
    submitted: list[object] = []

    class FakeJob:
        def __init__(self, **payload: object) -> None:
            constructed.append(payload)
            self.job_name: str | None = None

        def submit(self) -> str:
            self.job_name = "lm-mpi-job-global-safe"
            submitted.append(self)
            return f'Job "{self.job_name}" created.'

    monkeypatch.setitem(sys.modules, "client_lib", types.SimpleNamespace(Job=FakeJob))
    assert submit_global_job(config) == "lm-mpi-job-global-safe"
    assert len(constructed) == len(submitted) == 1
    assert constructed[0]["job_desc"] == JOB_DESC
    assert constructed[0]["queue_name"] == "shared"
    assert constructed[0]["priority_class"] == "shared-medium"
    assert checked_secret_paths == ["/home/jovyan/.comet.config"]


def test_submit_rejects_unacknowledged_job_without_claiming_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setattr(
        "experiments.global_cluster_submit._validate_runtime_secret_file",
        lambda _path: None,
    )

    class FailedJob:
        job_name = None

        def __init__(self, **payload: object) -> None:
            pass

        def submit(self) -> str:
            return "Error 503: untrusted-server-message"

    monkeypatch.setitem(sys.modules, "client_lib", types.SimpleNamespace(Job=FailedJob))
    with pytest.raises(GlobalClusterSubmissionError) as caught:
        submit_global_job(config)
    assert "untrusted-server-message" not in str(caught.value)


def test_submit_rejects_programmatic_comet_path_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _config(),
        execution={
            **_config().execution,
            "comet_config_file": "/tmp/not-the-approved-comet-config",
        },
    )
    monkeypatch.setattr(
        "experiments.global_cluster_submit._validate_runtime_secret_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("must fail first")),
    )

    with pytest.raises(GlobalClusterConfigError, match="comet_config_file"):
        submit_global_job(config)


def test_submit_cli_prints_only_one_safe_acknowledged_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[object] = []

    def fake_submit(config: object) -> str:
        calls.append(config)
        return "lm-mpi-job-one-global"

    monkeypatch.setattr(
        "experiments.global_cluster_submit.submit_global_job", fake_submit
    )
    assert main(["--config", str(CONFIG), "--submit"]) == 0
    output = capsys.readouterr().out
    assert len(calls) == 1
    assert "status: submitted" in output
    assert "job_count: 1" in output
    assert output.count("lm-mpi-job-one-global") == 1
    assert "COMET_API_KEY" not in output
