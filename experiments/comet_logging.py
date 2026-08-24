"""Small, secret-safe Comet adapter for pilot training runs.

The API key is deliberately never accepted as a function argument or config
field.  ``comet_ml`` reads it from the mode-0600 file selected by the public
``COMET_CONFIG`` path.  This keeps the credential out of Hydra's resolved
config, scheduler payloads, manifests and dry-run output.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Protocol


COMET_PROJECT_NAME = "lid-generalization"
COMET_WORKSPACE_NAME = "dont4rootme"
COMET_API_KEY_ENV = "COMET_API_KEY"
COMET_CONFIG_ENV = "COMET_CONFIG"
COMET_CONFIG_PATH = "/home/jovyan/.comet.config"
COMET_WORKSPACE_ENV = "COMET_WORKSPACE"
_FORBIDDEN_KEY_PARTS = ("api_key", "apikey", "password", "secret", "token")
_EXPERIMENT_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class CometConfigurationError(RuntimeError):
    """Raised before creating a Comet experiment with unsafe configuration."""


class _CometExperiment(Protocol):
    def set_name(self, name: str) -> Any: ...

    def add_tag(self, tag: str) -> Any: ...

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        prefix: str | None = None,
    ) -> Any: ...

    def log_parameters(
        self, parameters: Mapping[str, Any], *, prefix: str | None = None
    ) -> Any: ...

    def log_asset(self, file_data: str, *, file_name: str | None = None) -> Any: ...

    def end(self) -> Any: ...


def require_comet_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Validate public SDK routing without returning or copying the key.

    Production also verifies that the selected credential file is private.  An
    injected mapping is used by unit tests and is never treated as a filesystem
    capability.
    """

    source = os.environ if environ is None else environ
    required_names = {
        COMET_CONFIG_ENV: COMET_CONFIG_PATH,
        COMET_WORKSPACE_ENV: COMET_WORKSPACE_NAME,
        "COMET_PROJECT_NAME": COMET_PROJECT_NAME,
    }
    for variable, expected in required_names.items():
        configured = source.get(variable)
        if configured != expected:
            raise CometConfigurationError(
                f"{variable} must be exactly {expected!r}"
            )
    if "COMET_EXPERIMENT_NAME" in source:
        raise CometConfigurationError(
            "COMET_EXPERIMENT_NAME must be absent; Hydra owns the experiment name"
        )
    if environ is None:
        config_path = Path(COMET_CONFIG_PATH)
        try:
            mode = config_path.stat().st_mode
        except FileNotFoundError as error:
            raise CometConfigurationError(
                f"private Comet config does not exist: {config_path}"
            ) from error
        if not stat.S_ISREG(mode):
            raise CometConfigurationError(
                "private Comet config path must be a regular file"
            )
        if stat.S_IMODE(mode) & 0o077:
            raise CometConfigurationError(
                "private Comet config must have mode 0600 or stricter"
            )


def safe_scheduler_environment() -> dict[str, str]:
    """Return public Comet settings suitable for ``client_lib.Job``.

    In particular, this function never reads or returns ``COMET_API_KEY``.  It
    exposes only the path to the private runtime file so the Comet SDK can read
    the credential itself.
    """

    return {
        COMET_CONFIG_ENV: COMET_CONFIG_PATH,
        COMET_WORKSPACE_ENV: COMET_WORKSPACE_NAME,
        "COMET_PROJECT_NAME": COMET_PROJECT_NAME,
        "COMET_MODE": "online",
        "COMET_LOGGING_CONSOLE": "true",
    }


def _is_forbidden_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(part in normalized for part in _FORBIDDEN_KEY_PARTS)


def _assert_secret_free(value: object, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_key(key):
                raise CometConfigurationError(
                    f"secret-like field is forbidden in Comet events: {path}.{key}"
                )
            _assert_secret_free(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_secret_free(child, path=f"{path}[{index}]")


def _flatten(
    value: Mapping[str, Any], *, prefix: str = ""
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            flattened.update(_flatten(child, prefix=name))
        elif isinstance(child, (list, tuple)):
            flattened[name] = json.dumps(child, sort_keys=True, default=str)
        elif isinstance(child, Path):
            flattened[name] = str(child)
        else:
            flattened[name] = child
    return flattened


def _validate_experiment_name(experiment_name: str) -> str:
    if (
        not isinstance(experiment_name, str)
        or not _EXPERIMENT_NAME.fullmatch(experiment_name)
        or experiment_name.startswith("ent-block-")
        or experiment_name.endswith("-eval")
    ):
        raise CometConfigurationError(
            "Comet experiment name must be a model-specific lowercase "
            "kebab-case identifier without legacy affixes"
        )
    return experiment_name


class CometEventLogger:
    """Callable adapter for ``run_pilot(..., log_callback=...)``."""

    def __init__(
        self, experiment: _CometExperiment, *, experiment_name: str
    ) -> None:
        self._experiment = experiment
        self._experiment_name = experiment_name

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(project={COMET_PROJECT_NAME!r}, "
            f"experiment={self._experiment_name!r})"
        )

    def __call__(self, event_name: str, payload: Mapping[str, Any]) -> None:
        if not isinstance(event_name, str) or not event_name.strip():
            raise CometConfigurationError("Comet event name must be non-empty")
        if not isinstance(payload, Mapping):
            raise CometConfigurationError("Comet event payload must be a mapping")
        _assert_secret_free(payload)

        flattened = _flatten(payload)
        raw_step = flattened.pop("step", None)
        step: int | None = None
        if raw_step is not None:
            if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
                raise CometConfigurationError("event step must be numeric")
            step = int(raw_step)

        metrics: dict[str, float] = {}
        parameters: dict[str, Any] = {}
        for key, value in flattened.items():
            if isinstance(value, bool):
                parameters[key] = value
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[key] = float(value)
            else:
                parameters[key] = value

        if metrics:
            self._experiment.log_metrics(
                metrics,
                step=step,
                prefix=event_name,
            )
        if parameters:
            self._experiment.log_parameters(parameters, prefix=event_name)

    def log_asset(self, path: str | Path, *, name: str | None = None) -> None:
        asset = Path(path)
        if not asset.is_file():
            raise FileNotFoundError(asset)
        self._experiment.log_asset(str(asset), file_name=name)

    def end(self) -> None:
        self._experiment.end()


def create_comet_event_logger(
    *,
    experiment_name: str,
    tags: tuple[str, ...] = (),
    environ: Mapping[str, str] | None = None,
) -> CometEventLogger:
    """Create one Comet run in the approved namespace.

    ``environ`` exists for validation/testing only.  The SDK itself reads the
    real process environment, so a caller cannot pass the credential through
    this API.
    """

    require_comet_environment(environ)
    experiment_name = _validate_experiment_name(experiment_name)
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            raise CometConfigurationError("Comet tags must be non-empty strings")
    comet_ml = importlib.import_module("comet_ml")
    experiment = comet_ml.Experiment(
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE_NAME,
    )
    experiment.set_name(experiment_name)
    experiment.add_tag(experiment_name)
    for tag in tags:
        experiment.add_tag(tag)
    return CometEventLogger(experiment, experiment_name=experiment_name)


def create_comet_callback(
    *,
    experiment_name: str,
    tags: tuple[str, ...] = (),
    environ: Mapping[str, str] | None = None,
) -> tuple[
    Callable[[str, Mapping[str, Any]], None],
    Callable[[], None],
]:
    """Return the callback/close pair consumed by the pilot runner."""

    logger = create_comet_event_logger(
        experiment_name=experiment_name,
        tags=tags,
        environ=environ,
    )
    return logger, logger.end


__all__ = [
    "COMET_API_KEY_ENV",
    "COMET_CONFIG_ENV",
    "COMET_CONFIG_PATH",
    "COMET_PROJECT_NAME",
    "COMET_WORKSPACE_ENV",
    "COMET_WORKSPACE_NAME",
    "CometConfigurationError",
    "CometEventLogger",
    "create_comet_callback",
    "create_comet_event_logger",
    "require_comet_environment",
    "safe_scheduler_environment",
]
