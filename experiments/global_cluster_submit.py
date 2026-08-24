"""Fail-closed one-job launcher for the resumable global GPU campaign.

The default action is a secret-free dry run.  A real submission requires the
explicit ``--submit`` flag and constructs exactly one ``client_lib.Job`` after
re-validating the complete scheduler payload.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shlex
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import OmegaConf

from experiments.cluster_submit import (
    CUBLAS_WORKSPACE_CONFIG,
    INSTANCE_TYPE,
    JOB_DESC,
    JOB_TYPE,
    N_GPUS,
    N_WORKERS,
    PREFLIGHT_CHECK,
    PRIORITY_CLASS,
    QUEUE_NAME,
    REGION,
)
from experiments.comet_logging import (
    COMET_API_KEY_ENV,
    COMET_CONFIG_ENV,
    COMET_CONFIG_PATH,
    COMET_PROJECT_NAME,
    COMET_WORKSPACE_NAME,
    safe_scheduler_environment,
)

REPO_ROOT = (
    "/home/jovyan/echimbulatov/fork_afedorov/constant_repos/"
    "lid-estimation-generalization"
)
BLOCK_DIFF_PYTHON = "/home/jovyan/.mlspace/envs/block-diff/bin/python"
GLOBAL_MODULE = "experiments.global_campaign"
GLOBAL_ENTRYPOINT = f"{REPO_ROOT}/experiments/global_job.py"
DATA_ROOT = f"{REPO_ROOT}/data/lid_benchmarks_exact/benchmarks"
GLOBAL_OUTPUT_ROOT = f"{REPO_ROOT}/artifacts/global"
GLOBAL_CAMPAIGN = "all_suites_all_models"
BASE_IMAGE = (
    "cr.ai.cloud.ru/2754eb6e-ae19-4123-87ce-06ec3cc96500/"
    "job-latentdiffusion:flash-clear"
)

_APPROVED_ENVIRONMENT: dict[str, str | int] = {
    **safe_scheduler_environment(),
    "PROJECT_ROOT": REPO_ROOT,
    "PYTHONPATH": REPO_ROOT,
    "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
    "MLS_JOB_TOTAL_GPU": N_GPUS,
    "MLS_JOB_REGION_NAME": REGION,
    "PYTHONNOUSERSITE": 1,
    "PIP_USER": "no",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    "LD_LIBRARY_PATH": (
        "/usr/local/nvidia/lib64:/usr/local/nvidia/lib:"
        "/usr/lib/x86_64-linux-gnu:/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib:"
        "/opt/hpcx/ucc/lib:/opt/hpcx/sharp/lib:"
        "/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/hcoll/lib:"
        "/usr/local/cuda-12.6/lib64"
    ),
    "NCCL_IB_HCA": (
        "mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_10:1,mlx5_11:1"
    ),
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NCCL_DEBUG": "INFO",
    "OMPI_MCA_coll_hcoll_enable": "0",
    "OMPI_MCA_btl": "^openib,smcuda",
    "OMPI_MCA_pml": "ucx",
}

_SAFE_SCRIPT = re.compile(r"[A-Za-z0-9_@%+=:,./ -]+\Z")
_SECRET_NAME_PARTS = ("api_key", "apikey", "password", "secret", "token")
_MODEL_DATASET_TOKENS = (
    "diffusion",
    "rectified",
    "affine_fm",
    "affine-flow",
    "flow_matching",
    "flow-matching",
    "schrodinger",
    "bridge",
    "normalizing_flow",
    "normalizing-flow",
    "gaussian4",
    "spaghetti",
    "sphere4",
    "arrows",
    "fmnist",
    "uniform_pca",
    "spiral_pca",
    "exp_pca",
    "crescent_moon",
)
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
        "campaign",
        "seed",
        "comet_config_file",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "scheduler", "execution", "environment"}
)


class GlobalClusterConfigError(ValueError):
    """Raised before any scheduler API call when global config is unsafe."""


class GlobalClusterSubmissionError(RuntimeError):
    """Raised when the scheduler does not safely acknowledge the one job."""


@dataclass(frozen=True)
class GlobalClusterConfig:
    scheduler: Mapping[str, Any]
    execution: Mapping[str, Any]
    environment: Mapping[str, str | int]


@dataclass(frozen=True)
class PlannedGlobalJob:
    campaign: str
    payload: Mapping[str, Any]


def _reject_unknown_fields(
    table: Mapping[str, Any], allowed: frozenset[str], *, path: str
) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise GlobalClusterConfigError(f"unknown fields in {path}: {sorted(unknown)}")


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GlobalClusterConfigError(f"{path} must be a mapping")
    return value


def _absolute_posix_path(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise GlobalClusterConfigError(f"{path} must be a non-empty string")
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise GlobalClusterConfigError(f"{path} must be an absolute normalized path")
    return str(parsed)


def _contains_secret_name(value: object) -> bool:
    normalized = str(value).lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_NAME_PARTS)


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
        "base_image": BASE_IMAGE,
    }
    for key, expected in required.items():
        actual = value.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise GlobalClusterConfigError(
                f"scheduler.{key} must be exactly {expected!r}"
            )
    base_image = value.get("base_image")
    if not isinstance(base_image, str) or not base_image.strip():
        raise GlobalClusterConfigError("scheduler.base_image must be non-empty")
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
        "module": GLOBAL_MODULE,
        "entrypoint": GLOBAL_ENTRYPOINT,
        "data_root": DATA_ROOT,
        "output_dir": GLOBAL_OUTPUT_ROOT,
        "campaign": GLOBAL_CAMPAIGN,
        "comet_config_file": COMET_CONFIG_PATH,
    }
    for key, expected in exact.items():
        if result.get(key) != expected:
            raise GlobalClusterConfigError(
                f"execution.{key} must be exactly {expected!r}"
            )
    seed = value.get("seed")
    if seed != 0 or isinstance(seed, bool):
        raise GlobalClusterConfigError("execution.seed must be exactly 0")
    return result


def _validate_environment(value: Mapping[str, Any]) -> dict[str, str | int]:
    result: dict[str, str | int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise GlobalClusterConfigError("environment keys must be non-empty strings")
        if _contains_secret_name(key):
            raise GlobalClusterConfigError(
                f"secret variable {key!r} must not enter scheduler env_variables"
            )
        if not isinstance(item, (str, int)) or isinstance(item, bool):
            raise GlobalClusterConfigError(
                f"environment value {key!r} must be a string or integer"
            )
        result[key] = item

    if "COMET_EXPERIMENT_NAME" in result:
        raise GlobalClusterConfigError(
            "environment.COMET_EXPERIMENT_NAME is forbidden; Hydra owns names"
        )
    if set(result) != set(_APPROVED_ENVIRONMENT):
        raise GlobalClusterConfigError(
            f"environment fields differ from the exact allowlist: {sorted(set(result))}"
        )
    for key, expected in _APPROVED_ENVIRONMENT.items():
        if result.get(key) != expected:
            raise GlobalClusterConfigError(f"environment.{key} must be {expected!r}")
    return result


def load_global_cluster_config(path: str | Path) -> GlobalClusterConfig:
    """Load and strictly validate the sole YAML source of job settings."""

    config_path = Path(path)
    if config_path.suffix != ".yaml":
        raise GlobalClusterConfigError("cluster config must use the .yaml extension")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    top = _require_mapping(raw, path="cluster config")
    _reject_unknown_fields(top, _TOP_LEVEL_FIELDS, path="cluster config")
    schema_version = top.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise GlobalClusterConfigError("cluster config schema_version must be 1")
    scheduler = _validate_scheduler(
        _require_mapping(top.get("scheduler"), path="scheduler")
    )
    execution = _validate_execution(
        _require_mapping(top.get("execution"), path="execution")
    )
    environment = _validate_environment(
        _require_mapping(top.get("environment"), path="environment")
    )
    return _validated_global_cluster_config(
        GlobalClusterConfig(
            scheduler=scheduler,
            execution=execution,
            environment=environment,
        )
    )


def _validated_global_cluster_config(
    config: GlobalClusterConfig,
) -> GlobalClusterConfig:
    """Revalidate even programmatically constructed or replaced configs."""

    scheduler = _validate_scheduler(config.scheduler)
    execution = _validate_execution(config.execution)
    environment = _validate_environment(config.environment)
    if environment["PROJECT_ROOT"] != execution["repo_root"]:
        raise GlobalClusterConfigError(
            "environment.PROJECT_ROOT must equal execution.repo_root"
        )
    if environment["PYTHONPATH"] != execution["repo_root"]:
        raise GlobalClusterConfigError(
            "environment.PYTHONPATH must equal execution.repo_root"
        )
    if environment[COMET_CONFIG_ENV] != execution["comet_config_file"]:
        raise GlobalClusterConfigError(
            f"environment.{COMET_CONFIG_ENV} must equal execution.comet_config_file"
        )
    return GlobalClusterConfig(
        scheduler=scheduler,
        execution=execution,
        environment=environment,
    )


def _job_script(config: GlobalClusterConfig) -> str:
    execution = config.execution
    arguments = [
        str(execution["entrypoint"]),
        f"campaign={execution['campaign']}",
        f"data.root={execution['data_root']}",
        f"output_root={execution['output_dir']}",
        f"seed={execution['seed']}",
        f"logging.project={COMET_PROJECT_NAME}",
        f"logging.workspace={COMET_WORKSPACE_NAME}",
    ]
    return shlex.join(arguments)


def build_global_job_payload(config: GlobalClusterConfig) -> dict[str, Any]:
    """Build the sole scheduler payload for the complete campaign."""

    config = _validated_global_cluster_config(config)
    scheduler = config.scheduler
    payload: dict[str, Any] = {
        "job_desc": scheduler["job_desc"],
        "queue_name": scheduler["queue_name"],
        "base_image": scheduler["base_image"],
        "script": _job_script(config),
        "n_workers": scheduler["n_workers"],
        "instance_type": scheduler["instance_type"],
        "type": scheduler["type"],
        "preflight_check": scheduler["preflight_check"],
        "env_variables": dict(config.environment),
        "region": scheduler["region"],
        "flags": {},
        "priority_class": scheduler["priority_class"],
    }
    validate_global_job_payload(payload)
    return payload


def validate_global_job_payload(payload: Mapping[str, Any]) -> None:
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
        raise GlobalClusterConfigError(
            f"job payload fields differ from allowlist: {sorted(set(payload))}"
        )
    exact = {
        "job_desc": JOB_DESC,
        "queue_name": QUEUE_NAME,
        "base_image": BASE_IMAGE,
        "n_workers": N_WORKERS,
        "instance_type": INSTANCE_TYPE,
        "type": JOB_TYPE,
        "preflight_check": PREFLIGHT_CHECK,
        "region": REGION,
        "priority_class": PRIORITY_CLASS,
        "flags": {},
    }
    for key, expected in exact.items():
        actual = payload.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise GlobalClusterConfigError(
                f"payload.{key} must be exactly {expected!r}"
            )

    environment = _require_mapping(payload.get("env_variables"), path="env_variables")
    _validate_environment(environment)
    if COMET_API_KEY_ENV in environment:
        raise GlobalClusterConfigError("COMET_API_KEY must never be serialized")

    # Script and the legacy fixed job description are the only scheduler
    # fields exempted from identity scanning.  The script selects one global
    # Hydra campaign but contains no individual model or dataset name.
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"script", "job_desc", "base_image"}
    }
    for item in _walk_strings(metadata):
        lowered = item.lower()
        if any(token in lowered for token in _MODEL_DATASET_TOKENS):
            raise GlobalClusterConfigError(
                "model/dataset identity is forbidden in scheduler metadata"
            )

    script = payload.get("script")
    if not isinstance(script, str) or not script:
        raise GlobalClusterConfigError("job script must be a non-empty string")
    if not _SAFE_SCRIPT.fullmatch(script):
        raise GlobalClusterConfigError(
            "job script must be one safe command without shell operators"
        )
    command = shlex.split(script)
    expected_arguments = [
        GLOBAL_ENTRYPOINT,
        f"campaign={GLOBAL_CAMPAIGN}",
        f"data.root={DATA_ROOT}",
        f"output_root={GLOBAL_OUTPUT_ROOT}",
        "seed=0",
        f"logging.project={COMET_PROJECT_NAME}",
        f"logging.workspace={COMET_WORKSPACE_NAME}",
    ]
    if command != expected_arguments:
        raise GlobalClusterConfigError(
            "job script Hydra overrides differ from the global allowlist"
        )
    if "COMET_" in script:
        raise GlobalClusterConfigError(
            "job script must not handle Comet environment variables"
        )


def plan_global_job(config: GlobalClusterConfig) -> PlannedGlobalJob:
    """Return exactly one validated job plan."""

    return PlannedGlobalJob(
        campaign=GLOBAL_CAMPAIGN,
        payload=build_global_job_payload(config),
    )


def _validate_runtime_secret_file(path: str) -> None:
    """Check the private Comet config without opening or reading it."""

    secret_path = Path(path)
    try:
        mode = secret_path.stat().st_mode
    except FileNotFoundError as error:
        raise GlobalClusterConfigError(
            f"private Comet env file does not exist: {secret_path}"
        ) from error
    if not stat.S_ISREG(mode):
        raise GlobalClusterConfigError("private Comet env path must be a regular file")
    if stat.S_IMODE(mode) & 0o077:
        raise GlobalClusterConfigError(
            "private Comet env file must have mode 0600 or stricter"
        )


def _validated_submission_name(job: Any, response: Any) -> str:
    name = getattr(job, "job_name", None)
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or len(name) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise GlobalClusterSubmissionError(
            "cluster did not return a safe non-empty job name"
        )
    if response != f'Job "{name}" created.':
        raise GlobalClusterSubmissionError(
            "cluster did not acknowledge global job creation"
        )
    return name


def submit_global_job(config: GlobalClusterConfig) -> str:
    """Submit exactly one job after all local and payload checks pass."""

    planned = plan_global_job(config)
    environment = _require_mapping(
        planned.payload.get("env_variables"), path="env_variables"
    )
    _validate_runtime_secret_file(str(environment[COMET_CONFIG_ENV]))
    validate_global_job_payload(planned.payload)
    client_lib = importlib.import_module("client_lib")
    job = client_lib.Job(**planned.payload)
    try:
        response = job.submit()
    except Exception as error:
        raise GlobalClusterSubmissionError(
            "global job raised during submission"
        ) from error
    return _validated_submission_name(job, response)


def _dry_run_document(job: PlannedGlobalJob) -> str:
    document = {
        "schema_version": 1,
        "mode": "dry-run",
        "job_count": 1,
        "campaign": job.campaign,
        "payload": dict(job.payload),
    }
    return OmegaConf.to_yaml(OmegaConf.create(document), resolve=True, sort_keys=True)


def _default_config_path() -> Path:
    module_file = Path(__file__).resolve()
    candidates = (
        module_file.parent / "configs" / "cluster" / "shared_a100_global.yaml",
        module_file.parents[1] / "configs" / "cluster" / "shared_a100_global.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit the one global job after validation (default: dry run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_global_cluster_config(args.config)
    if args.submit:
        job_name = submit_global_job(config)
        print(
            OmegaConf.to_yaml(
                OmegaConf.create(
                    {
                        "status": "submitted",
                        "project": COMET_PROJECT_NAME,
                        "campaign": GLOBAL_CAMPAIGN,
                        "job_count": 1,
                        "job_name": job_name,
                    }
                ),
                resolve=True,
                sort_keys=True,
            ),
            end="",
        )
    else:
        print(_dry_run_document(plan_global_job(config)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BASE_IMAGE",
    "BLOCK_DIFF_PYTHON",
    "DATA_ROOT",
    "GLOBAL_CAMPAIGN",
    "GLOBAL_ENTRYPOINT",
    "GLOBAL_MODULE",
    "GLOBAL_OUTPUT_ROOT",
    "GlobalClusterConfig",
    "GlobalClusterConfigError",
    "GlobalClusterSubmissionError",
    "PlannedGlobalJob",
    "build_global_job_payload",
    "load_global_cluster_config",
    "main",
    "plan_global_job",
    "submit_global_job",
    "validate_global_job_payload",
]
