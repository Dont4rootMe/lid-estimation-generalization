"""Fail-closed launcher for the two approved A100 pilot jobs.

The default action is a dry run.  Submission requires an explicit ``--submit``
flag and re-validates every scheduler field immediately before constructing a
``client_lib.Job``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import sys
from typing import Any

from omegaconf import OmegaConf

from experiments.comet_logging import (
    COMET_API_KEY_ENV,
    COMET_CONFIG_ENV,
    COMET_CONFIG_PATH,
    COMET_PROJECT_NAME,
    COMET_WORKSPACE_NAME,
    safe_scheduler_environment,
)


JOB_DESC = "echimbulatov | ent-block-diffusion-eval #ID0137 #rnd"
QUEUE_NAME = "shared"
PRIORITY_CLASS = "shared-medium"
REGION = "A100-MT"
JOB_TYPE = "pytorch2"
PREFLIGHT_CHECK = True
N_WORKERS = 1
N_GPUS = 1
INSTANCE_TYPE = "a100.1gpu"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
APPROVED_FAMILIES = ("diffusion", "rectified_flow")
REPO_ROOT = (
    "/home/jovyan/echimbulatov/fork_afedorov/constant_repos/"
    "lid-estimation-generalization"
)
BLOCK_DIFF_PYTHON = "/home/jovyan/.mlspace/envs/block-diff/bin/python"
PILOT_MODULE = "experiments.pilot"
PILOT_ENTRYPOINT = f"{REPO_ROOT}/experiments/pilot_job.py"
DATA_ROOT = f"{REPO_ROOT}/data/lid_benchmarks_exact/benchmarks"
OUTPUT_ROOT = f"{REPO_ROOT}/artifacts/pilot"
_SAFE_SCRIPT = re.compile(r"[A-Za-z0-9_@%+=:,./ -]+\Z")

_SCHEDULER_FIELDS = frozenset(
    {
        "job_desc",
        "queue_name",
        "base_image",
        "n_workers",
        "n_gpus",
        "instance_type",
        "type",
        "preflight_check",
        "region",
        "priority_class",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "repo_root",
        "python",
        "module",
        "entrypoint",
        "data_root",
        "output_dir",
        "seed",
        "comet_config_file",
        "families",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "scheduler", "execution", "environment"}
)
_FORBIDDEN_METADATA_TOKENS = (
    "diffusion",
    "rectified_flow",
    "rectified-flow-matching",
    "affine_fm",
    "gaussian_diffusion",
    "e8_gaussian4_pca",
    "e8_spaghetti_pca",
    "e8_sphere4_pca",
)
_SECRET_NAME_PARTS = ("api_key", "apikey", "password", "secret", "token")


class ClusterConfigError(ValueError):
    """Raised before any scheduler API call when the payload is unsafe."""


class ClusterSubmissionError(RuntimeError):
    """Raised when the cluster did not acknowledge a submitted job.

    ``accepted_job_names`` is safe to surface to an operator and makes a
    partial two-job submission explicit.  Raw server responses are never
    retained because they are not trusted log material.
    """

    def __init__(self, message: str, *, accepted_job_names: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.accepted_job_names = tuple(accepted_job_names)


@dataclass(frozen=True)
class ClusterConfig:
    scheduler: Mapping[str, Any]
    execution: Mapping[str, Any]
    environment: Mapping[str, str | int]


@dataclass(frozen=True)
class PlannedJob:
    family: str
    experiment_name: str
    payload: Mapping[str, Any]


def _experiment_name_for_family(family: str) -> str:
    """Resolve the Comet name from the selected Hydra model group."""

    if family not in APPROVED_FAMILIES:
        raise ClusterConfigError(f"unapproved pilot family: {family!r}")
    from experiments.pilot import compose_pilot_config, validate_pilot_config

    try:
        resolved = validate_pilot_config(
            compose_pilot_config((f"pilot_model={family}", "seed=0"))
        )
    except Exception as error:
        raise ClusterConfigError(
            f"cannot resolve Hydra experiment name for family {family!r}"
        ) from error
    experiment_name = resolved.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ClusterConfigError(
            f"Hydra experiment name for family {family!r} must be non-empty"
        )
    return experiment_name


def _reject_unknown_fields(
    table: Mapping[str, Any], allowed: frozenset[str], *, path: str
) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ClusterConfigError(f"unknown fields in {path}: {sorted(unknown)}")


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClusterConfigError(f"{path} must be a mapping")
    return value


def _absolute_posix_path(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ClusterConfigError(f"{path} must be a non-empty string")
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise ClusterConfigError(f"{path} must be an absolute normalized path")
    return str(parsed)


def _contains_secret_name(value: object) -> bool:
    normalized = str(value).lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_NAME_PARTS)


def _validate_environment(value: Mapping[str, Any]) -> dict[str, str | int]:
    result: dict[str, str | int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ClusterConfigError("environment keys must be non-empty strings")
        if _contains_secret_name(key):
            raise ClusterConfigError(
                f"secret variable {key!r} must not enter scheduler env_variables"
            )
        if not isinstance(item, (str, int)) or isinstance(item, bool):
            raise ClusterConfigError(
                f"environment value {key!r} must be a string or integer"
            )
        result[key] = item

    if "COMET_EXPERIMENT_NAME" in result:
        raise ClusterConfigError(
            "environment.COMET_EXPERIMENT_NAME is forbidden; Hydra owns the name"
        )

    required_public = safe_scheduler_environment()
    for key, expected in required_public.items():
        if result.get(key) != expected:
            raise ClusterConfigError(f"environment.{key} must be {expected!r}")
    if result.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise ClusterConfigError(
            "environment.CUBLAS_WORKSPACE_CONFIG must be "
            f"{CUBLAS_WORKSPACE_CONFIG!r}"
        )
    if result.get("MLS_JOB_TOTAL_GPU") != N_GPUS:
        raise ClusterConfigError(f"environment.MLS_JOB_TOTAL_GPU must be {N_GPUS}")
    if result.get("MLS_JOB_REGION_NAME") != REGION:
        raise ClusterConfigError(
            f"environment.MLS_JOB_REGION_NAME must be {REGION!r}"
        )
    for key in ("PROJECT_ROOT", "PYTHONPATH"):
        if result.get(key) != REPO_ROOT:
            raise ClusterConfigError(f"environment.{key} must be {REPO_ROOT!r}")
    return result


def _validate_scheduler(value: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown_fields(value, _SCHEDULER_FIELDS, path="scheduler")
    required = {
        "job_desc": JOB_DESC,
        "queue_name": QUEUE_NAME,
        "n_workers": N_WORKERS,
        "n_gpus": N_GPUS,
        "instance_type": INSTANCE_TYPE,
        "type": JOB_TYPE,
        "preflight_check": PREFLIGHT_CHECK,
        "region": REGION,
        "priority_class": PRIORITY_CLASS,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ClusterConfigError(f"scheduler.{key} must be exactly {expected!r}")
    base_image = value.get("base_image")
    if not isinstance(base_image, str) or not base_image.strip():
        raise ClusterConfigError("scheduler.base_image must be a non-empty string")
    return dict(value)


def _validate_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown_fields(value, _EXECUTION_FIELDS, path="execution")
    result = dict(value)
    for key in (
        "repo_root",
        "python",
        "entrypoint",
        "data_root",
        "output_dir",
        "comet_config_file",
    ):
        result[key] = _absolute_posix_path(value.get(key), path=f"execution.{key}")
    exact = {
        "repo_root": REPO_ROOT,
        "python": BLOCK_DIFF_PYTHON,
        "module": PILOT_MODULE,
        "entrypoint": PILOT_ENTRYPOINT,
        "data_root": DATA_ROOT,
        "output_dir": OUTPUT_ROOT,
        "comet_config_file": COMET_CONFIG_PATH,
    }
    for key, expected in exact.items():
        if result.get(key) != expected:
            raise ClusterConfigError(f"execution.{key} must be exactly {expected!r}")
    seed = value.get("seed")
    if seed != 0 or isinstance(seed, bool):
        raise ClusterConfigError("execution.seed must be exactly 0")
    families = value.get("families")
    if not isinstance(families, list) or tuple(families) != APPROVED_FAMILIES:
        raise ClusterConfigError(
            f"execution.families must be exactly {list(APPROVED_FAMILIES)!r}"
        )
    return result


def load_cluster_config(path: str | Path) -> ClusterConfig:
    """Load and strictly validate the sole YAML source of launcher settings."""

    config_path = Path(path)
    if config_path.suffix != ".yaml":
        raise ClusterConfigError("cluster config must use the .yaml extension")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    top = _require_mapping(raw, path="cluster config")
    _reject_unknown_fields(top, _TOP_LEVEL_FIELDS, path="cluster config")
    if top.get("schema_version") != 1:
        raise ClusterConfigError("cluster config schema_version must be 1")
    scheduler = _validate_scheduler(
        _require_mapping(top.get("scheduler"), path="scheduler")
    )
    execution = _validate_execution(
        _require_mapping(top.get("execution"), path="execution")
    )
    environment = _validate_environment(
        _require_mapping(top.get("environment"), path="environment")
    )
    if environment.get("PROJECT_ROOT") != execution["repo_root"]:
        raise ClusterConfigError(
            "environment.PROJECT_ROOT must equal execution.repo_root"
        )
    if environment.get("PYTHONPATH") != execution["repo_root"]:
        raise ClusterConfigError(
            "environment.PYTHONPATH must equal execution.repo_root"
        )
    if environment.get(COMET_CONFIG_ENV) != execution["comet_config_file"]:
        raise ClusterConfigError(
            f"environment.{COMET_CONFIG_ENV} must equal execution.comet_config_file"
        )
    return ClusterConfig(
        scheduler=scheduler,
        execution=execution,
        environment=environment,
    )


def _job_script(config: ClusterConfig, family: str) -> str:
    execution = config.execution
    output_dir = str(PurePosixPath(str(execution["output_dir"])) / family)
    experiment_name = _experiment_name_for_family(family)
    python_args = [
        str(execution["entrypoint"]),
        f"pilot_model={family}",
        f"data.root={execution['data_root']}",
        f"output_root={output_dir}",
        f"seed={execution['seed']}",
        f"logging.project={COMET_PROJECT_NAME}",
        f"logging.experiment_name={experiment_name}",
        f"logging.workspace={COMET_WORKSPACE_NAME}",
    ]
    return shlex.join(python_args)


def build_job_payload(config: ClusterConfig, family: str) -> dict[str, Any]:
    if family not in APPROVED_FAMILIES:
        raise ClusterConfigError(f"unapproved pilot family: {family!r}")
    scheduler = config.scheduler
    payload: dict[str, Any] = {
        "job_desc": scheduler["job_desc"],
        "queue_name": scheduler["queue_name"],
        "base_image": scheduler["base_image"],
        "script": _job_script(config, family),
        "n_workers": scheduler["n_workers"],
        "instance_type": scheduler["instance_type"],
        "type": scheduler["type"],
        "preflight_check": scheduler["preflight_check"],
        "env_variables": dict(config.environment),
        "region": scheduler["region"],
        "flags": {},
        "priority_class": scheduler["priority_class"],
    }
    validate_job_payload(payload)
    return payload


def _walk_strings(value: object) -> Sequence[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            result.extend(_walk_strings(key))
            result.extend(_walk_strings(child))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for child in value:
            result.extend(_walk_strings(child))
        return result
    return [value] if isinstance(value, str) else []


def validate_job_payload(payload: Mapping[str, Any]) -> None:
    """Validate the exact object passed to ``client_lib.Job``."""

    expected_keys = {
        "job_desc",
        "queue_name",
        "base_image",
        "script",
        "n_workers",
        "instance_type",
        "type",
        "preflight_check",
        "env_variables",
        "region",
        "flags",
        "priority_class",
    }
    if set(payload) != expected_keys:
        raise ClusterConfigError(
            f"job payload fields differ from allowlist: {sorted(set(payload))}"
        )
    exact = {
        "job_desc": JOB_DESC,
        "queue_name": QUEUE_NAME,
        "n_workers": N_WORKERS,
        "instance_type": INSTANCE_TYPE,
        "type": JOB_TYPE,
        "preflight_check": PREFLIGHT_CHECK,
        "region": REGION,
        "priority_class": PRIORITY_CLASS,
        "flags": {},
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise ClusterConfigError(f"payload.{key} must be exactly {expected!r}")

    environment = _require_mapping(payload.get("env_variables"), path="env_variables")
    _validate_environment(environment)
    for key in environment:
        if _contains_secret_name(key):
            raise ClusterConfigError(f"secret {key!r} leaked into scheduler payload")
    if COMET_API_KEY_ENV in environment:
        raise ClusterConfigError("COMET_API_KEY must never be serialized")

    # The script necessarily selects a family.  All remaining scheduler-visible
    # metadata must stay family/dataset agnostic, apart from the fixed approved
    # namespace embedded verbatim in JOB_DESC and public Comet settings.
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"script", "job_desc", "base_image"}
    }
    for text in _walk_strings(metadata):
        lowered = text.lower()
        if any(token in lowered for token in _FORBIDDEN_METADATA_TOKENS):
            raise ClusterConfigError(
                "model/dataset identity is forbidden in scheduler metadata"
            )

    script = payload.get("script")
    if not isinstance(script, str) or not script:
        raise ClusterConfigError("job script must be a non-empty string")
    if not _SAFE_SCRIPT.fullmatch(script):
        raise ClusterConfigError(
            "job script must be one safe command without shell operators"
        )
    command = shlex.split(script)
    expected_prefix = [PILOT_ENTRYPOINT]
    if command[:1] != expected_prefix:
        raise ClusterConfigError(
            "job script must invoke the approved Python source entrypoint"
        )
    if len(command) != 8:
        raise ClusterConfigError("job script must contain exactly seven Hydra overrides")
    family_argument = command[1]
    if not family_argument.startswith("pilot_model="):
        raise ClusterConfigError("job script must select one approved model family")
    family = family_argument.removeprefix("pilot_model=")
    if family not in APPROVED_FAMILIES:
        raise ClusterConfigError("job script selects an unapproved model family")
    experiment_name = _experiment_name_for_family(family)
    expected_arguments = [
        f"pilot_model={family}",
        f"data.root={DATA_ROOT}",
        f"output_root={OUTPUT_ROOT}/{family}",
        "seed=0",
        f"logging.project={COMET_PROJECT_NAME}",
        f"logging.experiment_name={experiment_name}",
        f"logging.workspace={COMET_WORKSPACE_NAME}",
    ]
    if command[1:] != expected_arguments:
        raise ClusterConfigError("job script Hydra overrides differ from the allowlist")
    if f"logging.project={COMET_PROJECT_NAME}" not in script:
        raise ClusterConfigError("job script must pin the approved Comet project")
    if f"logging.experiment_name={experiment_name}" not in script:
        raise ClusterConfigError(
            "job script must pin the approved Comet experiment name"
        )
    if f"logging.workspace={COMET_WORKSPACE_NAME}" not in script:
        raise ClusterConfigError("job script must pin the approved Comet workspace")
    if "COMET_" in script:
        raise ClusterConfigError("job script must not handle Comet environment values")


def _selected_families(family: str | None) -> tuple[str, ...]:
    if family is None:
        return APPROVED_FAMILIES
    if family not in APPROVED_FAMILIES:
        raise ClusterConfigError(f"unapproved pilot family: {family!r}")
    return (family,)


def plan_jobs(
    config: ClusterConfig, family: str | None = None
) -> tuple[PlannedJob, ...]:
    return tuple(
        PlannedJob(
            family=selected,
            experiment_name=_experiment_name_for_family(selected),
            payload=build_job_payload(config, selected),
        )
        for selected in _selected_families(family)
    )


def _validate_runtime_secret_file(path: str) -> None:
    """Check the secret source without opening or reading its contents."""

    secret_path = Path(path)
    try:
        mode = secret_path.stat().st_mode
    except FileNotFoundError as error:
        raise ClusterConfigError(
            f"private Comet env file does not exist: {secret_path}"
        ) from error
    if not stat.S_ISREG(mode):
        raise ClusterConfigError("private Comet env path must be a regular file")
    if stat.S_IMODE(mode) & 0o077:
        raise ClusterConfigError("private Comet env file must have mode 0600 or stricter")


def _validated_submission_name(job: Any, response: Any) -> str:
    name = getattr(job, "job_name", None)
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or len(name) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ClusterSubmissionError("cluster did not return a safe non-empty job name")
    expected = f'Job "{name}" created.'
    if response != expected:
        raise ClusterSubmissionError("cluster did not acknowledge job creation")
    return name


def submit_jobs(
    config: ClusterConfig, family: str | None = None
) -> tuple[str, ...]:
    """Submit the selected jobs only after all local and payload checks pass."""

    jobs = plan_jobs(config, family)
    _validate_runtime_secret_file(str(config.execution["comet_config_file"]))
    client_lib = importlib.import_module("client_lib")
    scheduler_jobs: list[Any] = []
    for planned in jobs:
        validate_job_payload(planned.payload)
        scheduler_jobs.append(client_lib.Job(**planned.payload))
    accepted: list[str] = []
    for index, job in enumerate(scheduler_jobs, start=1):
        try:
            response = job.submit()
        except Exception as error:
            raise ClusterSubmissionError(
                f"job {index} of {len(scheduler_jobs)} raised during submission; "
                f"{len(accepted)} job(s) were already accepted",
                accepted_job_names=accepted,
            ) from error
        try:
            accepted.append(_validated_submission_name(job, response))
        except ClusterSubmissionError as error:
            raise ClusterSubmissionError(
                f"job {index} of {len(scheduler_jobs)} was not accepted; "
                f"{len(accepted)} job(s) were already accepted",
                accepted_job_names=accepted,
            ) from error
    return tuple(accepted)


def _dry_run_document(jobs: Sequence[PlannedJob]) -> str:
    document = {
        "schema_version": 1,
        "mode": "dry-run",
        "jobs": [
            {
                "family": planned.family,
                "experiment_name": planned.experiment_name,
                "payload": dict(planned.payload),
            }
            for planned in jobs
        ],
    }
    return OmegaConf.to_yaml(OmegaConf.create(document), resolve=True, sort_keys=True)


def _default_config_path() -> Path:
    module_file = Path(__file__).resolve()
    candidates = (
        module_file.parent / "configs" / "cluster" / "shared_a100.yaml",
        module_file.parents[1] / "configs" / "cluster" / "shared_a100.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument(
        "--family",
        choices=APPROVED_FAMILIES,
        help="plan or submit only one approved family (default: both)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit after validation (default: print a secret-free dry run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_cluster_config(args.config)
    selected_families = _selected_families(args.family)
    if args.submit:
        job_names = submit_jobs(config, args.family)
        print(
            OmegaConf.to_yaml(
                OmegaConf.create(
                    {
                        "status": "submitted",
                        "project": COMET_PROJECT_NAME,
                        "experiment_names": {
                            family: _experiment_name_for_family(family)
                            for family in selected_families
                        },
                        "job_names": list(job_names),
                    }
                ),
                resolve=True,
                sort_keys=True,
            ),
            end="",
        )
    else:
        print(_dry_run_document(plan_jobs(config, args.family)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "APPROVED_FAMILIES",
    "CUBLAS_WORKSPACE_CONFIG",
    "ClusterConfig",
    "ClusterConfigError",
    "ClusterSubmissionError",
    "INSTANCE_TYPE",
    "JOB_DESC",
    "PlannedJob",
    "build_job_payload",
    "load_cluster_config",
    "main",
    "plan_jobs",
    "submit_jobs",
    "validate_job_payload",
]
