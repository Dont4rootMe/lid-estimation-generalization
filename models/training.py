"""Deterministic training and inference for the pilot neural-field families.

Two objectives are implemented:

* variance-exploding diffusion with an ``x0`` denoiser parameterization;
* linear rectified-flow matching from Gaussian noise to the data.

The diffusion denoiser is converted to a score only at inference time.  This
keeps training and divergence estimation numerically stable while preserving
the denoising-score-matching identity
``score(y, sigma) = (x0_hat(y, sigma) - y) / sigma**2``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from models.neural_fields import (
    NeuralFieldConfig,
    ScaleConditionedNeuralField,
    exact_divergence,
    hutchinson_divergence,
)
from models.readouts import (
    diffusion_flipd,
    rectified_flow_full,
    rectified_flow_response,
)


Family = Literal["gaussian_diffusion", "rectified_flow"]
LogCallback = Callable[[Mapping[str, float | int | bool | str]], None]
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Hydra-friendly scalar settings for one model/dataset training run.

    ``num_workers`` is retained in the serialized experiment schema for
    launcher compatibility.  Training receives already materialized arrays
    and moves them to one contiguous tensor, so it intentionally creates no
    data-loader worker processes.
    """

    seed: int = 0
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_dim: int = 256
    depth: int = 4
    time_embedding_dim: int = 64
    validation_interval: int = 1
    early_stopping_patience: int | None = 20
    gradient_clip_norm: float | None = 1.0
    num_workers: int = 0
    deterministic: bool = True
    sigma_min: float = 0.02
    sigma_max: float = 1.0
    time_min: float = 1.0e-3
    time_max: float = 1.0 - 1.0e-3
    fourier_features: int = 32
    max_condition_frequency: float = 100.0
    dropout: float = 0.0
    normalize: bool = True
    normalization_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        integer_positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "time_embedding_dim": self.time_embedding_dim,
            "validation_interval": self.validation_interval,
            "fourier_features": self.fourier_features,
        }
        for name, value in integer_positive.items():
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be null or positive")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be null or positive")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError("sigma_min and sigma_max must satisfy 0 < min < max")
        if not 0 <= self.time_min < self.time_max <= 1:
            raise ValueError("time bounds must satisfy 0 <= min < max <= 1")
        if (
            not math.isfinite(self.max_condition_frequency)
            or self.max_condition_frequency < 1
        ):
            raise ValueError("max_condition_frequency must be finite and >= 1")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be boolean")
        if (
            not math.isfinite(self.normalization_epsilon)
            or self.normalization_epsilon <= 0
        ):
            raise ValueError("normalization_epsilon must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown training settings: {sorted(unknown)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class FieldPrediction:
    """Primitive field values in model coordinates for one readout scale."""

    field: npt.NDArray[np.float64]
    divergence: npt.NDArray[np.float64]
    evaluation_point: npt.NDArray[np.float64]
    condition: float


@dataclass(frozen=True)
class TrainingResult:
    family: Family
    model: ScaleConditionedNeuralField
    config: TrainingConfig
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str
    normalization_mean: Tensor
    normalization_scale: float
    preprocessing: Mapping[str, str | float | int]
    preprocessing_sha256: str

    @property
    def metrics(self) -> dict[str, float | int]:
        final = self.history[-1]
        return {
            "epochs_completed": final.epoch,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "final_train_loss": final.train_loss,
            "final_validation_loss": final.validation_loss,
        }


def _canonical_family(family: str) -> Family:
    aliases: dict[str, Family] = {
        "diffusion": "gaussian_diffusion",
        "gaussian_diffusion": "gaussian_diffusion",
        "rectified_flow": "rectified_flow",
    }
    try:
        return aliases[str(family)]
    except KeyError as exc:
        raise ValueError(
            "family must be diffusion, gaussian_diffusion, or rectified_flow"
        ) from exc


def _coerce_config(config: TrainingConfig | Mapping[str, Any]) -> TrainingConfig:
    if isinstance(config, TrainingConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be TrainingConfig or a mapping")
    return TrainingConfig.from_mapping(config)


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch before model initialization."""

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flat_finite_data(value: Any, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim < 2 or tensor.shape[0] <= 0:
        raise ValueError(f"{name} must have shape (nonempty batch, ...)")
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(
        tensor.shape[0], -1
    )
    if tensor.shape[1] <= 0:
        raise ValueError(f"{name} has no feature dimensions")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


def _mean_sha256(mean: Tensor) -> str:
    array = np.ascontiguousarray(mean.detach().cpu().numpy().astype("<f4", copy=False))
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _preprocessing_identity(
    mean: Tensor, scale: float, *, normalized: bool
) -> tuple[dict[str, str | float | int], str]:
    preprocessing: dict[str, str | float | int] = {
        "schema_version": 1,
        "kind": "train_mean_global_rms_v1" if normalized else "identity_v1",
        "ambient_dim": mean.numel(),
        "mean_sha256": _mean_sha256(mean),
        "scalar_scale": float(scale),
    }
    canonical = json.dumps(
        preprocessing,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return preprocessing, hashlib.sha256(canonical).hexdigest()


def _normalization(
    train: Tensor, *, enabled: bool, epsilon: float
) -> tuple[Tensor, float, dict[str, str | float | int], str]:
    if enabled:
        mean = train.mean(dim=0)
        rms = torch.sqrt(torch.mean((train - mean).square()))
        scale = float(rms.item())
        if not math.isfinite(scale) or scale < epsilon:
            scale = 1.0
    else:
        mean = torch.zeros(train.shape[1], dtype=train.dtype)
        scale = 1.0
    preprocessing, preprocessing_sha256 = _preprocessing_identity(
        mean, scale, normalized=enabled
    )
    return mean.contiguous(), scale, preprocessing, preprocessing_sha256


def _sample_log_uniform(
    batch_size: int,
    *,
    minimum: float,
    maximum: float,
    data: Tensor,
    generator: torch.Generator,
) -> Tensor:
    uniform = torch.rand(
        batch_size,
        device=data.device,
        dtype=data.dtype,
        generator=generator,
    )
    return torch.exp(
        math.log(minimum) + uniform * (math.log(maximum) - math.log(minimum))
    )


def diffusion_ve_dsm_loss(
    model: nn.Module,
    clean: Tensor,
    *,
    sigma_min: float,
    sigma_max: float,
    generator: torch.Generator,
) -> Tensor:
    """VE denoising-score-matching loss in a stable ``x0`` parameterization."""

    sigma = _sample_log_uniform(
        clean.shape[0],
        minimum=sigma_min,
        maximum=sigma_max,
        data=clean,
        generator=generator,
    )
    noise = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    sigma_broadcast = sigma.reshape(-1, *([1] * (clean.ndim - 1)))
    perturbed = clean + sigma_broadcast * noise
    denoised = model(perturbed, sigma)
    # Equivalent to sigma^2 * ||score + noise/sigma||^2 after substituting
    # score=(denoised-perturbed)/sigma^2.
    return ((denoised - clean) / sigma_broadcast).square().flatten(1).mean()


def rectified_flow_matching_loss(
    model: nn.Module,
    data: Tensor,
    *,
    time_min: float,
    time_max: float,
    generator: torch.Generator,
) -> Tensor:
    """Linear flow-matching objective for ``z -> data`` transport."""

    time = time_min + (time_max - time_min) * torch.rand(
        data.shape[0],
        device=data.device,
        dtype=data.dtype,
        generator=generator,
    )
    noise = torch.randn(
        data.shape, device=data.device, dtype=data.dtype, generator=generator
    )
    time_broadcast = time.reshape(-1, *([1] * (data.ndim - 1)))
    interpolated = time_broadcast * data + (1.0 - time_broadcast) * noise
    target_velocity = data - noise
    velocity = model(interpolated, time)
    return (velocity - target_velocity).square().flatten(1).mean()


def _objective(
    family: Family,
    model: nn.Module,
    batch: Tensor,
    config: TrainingConfig,
    generator: torch.Generator,
) -> Tensor:
    if family == "gaussian_diffusion":
        return diffusion_ve_dsm_loss(
            model,
            batch,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            generator=generator,
        )
    return rectified_flow_matching_loss(
        model,
        batch,
        time_min=config.time_min,
        time_max=config.time_max,
        generator=generator,
    )


def _validation_loss(
    family: Family,
    model: nn.Module,
    validation: Tensor,
    config: TrainingConfig,
    *,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=validation.device)
    generator.manual_seed(seed)
    weighted_loss = 0.0
    with torch.no_grad():
        for start in range(0, validation.shape[0], config.batch_size):
            batch = validation[start : start + config.batch_size]
            loss = _objective(family, model, batch, config, generator)
            weighted_loss += float(loss.item()) * batch.shape[0]
    return weighted_loss / validation.shape[0]


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }


def _save_checkpoint(
    path: Path,
    *,
    family: Family,
    model: ScaleConditionedNeuralField,
    config: TrainingConfig,
    history: tuple[EpochMetrics, ...],
    best_epoch: int,
    best_validation_loss: float,
    normalization_mean: Tensor,
    normalization_scale: float,
    preprocessing: Mapping[str, str | float | int],
    preprocessing_sha256: str,
) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": family,
        "architecture": model.config.to_dict(),
        "training_config": config.to_dict(),
        "model_state": _cpu_state_dict(model),
        "history": [metric.to_dict() for metric in history],
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "normalization": {
            "mean": normalization_mean.detach().cpu().contiguous(),
            "scale": normalization_scale,
            "preprocessing": dict(preprocessing),
            "sha256": preprocessing_sha256,
        },
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return _checkpoint_sha256(path)


def train_model(
    family: str,
    train: Any,
    validation: Any,
    config: TrainingConfig | Mapping[str, Any],
    checkpoint_path: str | Path,
    log_callback: LogCallback | None = None,
) -> TrainingResult:
    """Train one family on one dataset and atomically write its best checkpoint."""

    canonical_family = _canonical_family(family)
    resolved_config = _coerce_config(config)
    train_cpu = _flat_finite_data(train, name="train")
    validation_cpu = _flat_finite_data(validation, name="validation")
    if train_cpu.shape[1] != validation_cpu.shape[1]:
        raise ValueError("train and validation ambient dimensions do not match")
    mean, scale, preprocessing, preprocessing_sha256 = _normalization(
        train_cpu,
        enabled=resolved_config.normalize,
        epsilon=resolved_config.normalization_epsilon,
    )
    train_cpu = (train_cpu - mean) / scale
    validation_cpu = (validation_cpu - mean) / scale
    device = _resolve_device(resolved_config.device)
    seed_everything(resolved_config.seed)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    if resolved_config.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        architecture = NeuralFieldConfig(
            ambient_dim=train_cpu.shape[1],
            hidden_dim=resolved_config.hidden_dim,
            depth=resolved_config.depth,
            condition_dim=resolved_config.time_embedding_dim,
            fourier_features=resolved_config.fourier_features,
            max_condition_frequency=resolved_config.max_condition_frequency,
            dropout=resolved_config.dropout,
            condition_transform=(
                "log" if canonical_family == "gaussian_diffusion" else "linear"
            ),
        )
        model = ScaleConditionedNeuralField(architecture).to(device)
        setattr(model, "_lid_family", canonical_family)
        train_tensor = train_cpu.to(device)
        validation_tensor = validation_cpu.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved_config.learning_rate,
            weight_decay=resolved_config.weight_decay,
        )
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(resolved_config.seed + 17)
        objective_generator = torch.Generator(device=device)
        objective_generator.manual_seed(resolved_config.seed + 31)

        history_list: list[EpochMetrics] = []
        best_validation_loss = math.inf
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None
        stale_validations = 0
        for epoch in range(1, resolved_config.epochs + 1):
            model.train()
            permutation = torch.randperm(
                train_tensor.shape[0], generator=shuffle_generator
            )
            accumulated = 0.0
            for start in range(0, train_tensor.shape[0], resolved_config.batch_size):
                indices = permutation[start : start + resolved_config.batch_size].to(
                    device
                )
                batch = train_tensor[indices]
                optimizer.zero_grad(set_to_none=True)
                loss = _objective(
                    canonical_family,
                    model,
                    batch,
                    resolved_config,
                    objective_generator,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch}"
                    )
                loss.backward()
                if resolved_config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), resolved_config.gradient_clip_norm
                    )
                optimizer.step()
                accumulated += float(loss.item()) * batch.shape[0]
            train_loss = accumulated / train_tensor.shape[0]

            should_validate = (
                epoch % resolved_config.validation_interval == 0
                or epoch == resolved_config.epochs
            )
            if not should_validate:
                if log_callback is not None:
                    log_callback(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "validated": False,
                        }
                    )
                continue
            validation_loss = _validation_loss(
                canonical_family,
                model,
                validation_tensor,
                resolved_config,
                seed=resolved_config.seed + 1_000_003,
            )
            if not math.isfinite(validation_loss):
                raise FloatingPointError(
                    f"non-finite validation loss at epoch {epoch}"
                )
            metric = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
            history_list.append(metric)
            improved = validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = _cpu_state_dict(model)
                stale_validations = 0
            else:
                stale_validations += 1
            if log_callback is not None:
                log_callback(
                    {
                        **metric.to_dict(),
                        "best_validation_loss": best_validation_loss,
                        "validated": True,
                    }
                )
            if (
                resolved_config.early_stopping_patience is not None
                and stale_validations >= resolved_config.early_stopping_patience
            ):
                break
        if best_state is None or not history_list:
            raise RuntimeError("training ended without a validation measurement")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        history = tuple(history_list)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        checkpoint_sha256 = _save_checkpoint(
            checkpoint,
            family=canonical_family,
            model=model,
            config=resolved_config,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            normalization_mean=mean,
            normalization_scale=scale,
            preprocessing=preprocessing,
            preprocessing_sha256=preprocessing_sha256,
        )
        return TrainingResult(
            family=canonical_family,
            model=model,
            config=resolved_config,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            normalization_mean=mean.detach().cpu().contiguous(),
            normalization_scale=scale,
            preprocessing=preprocessing,
            preprocessing_sha256=preprocessing_sha256,
        )
    finally:
        if resolved_config.deterministic:
            torch.use_deterministic_algorithms(previous_deterministic)
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic


def load_checkpoint(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> TrainingResult:
    """Load a portable weights-only checkpoint and validate its preprocessing."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_device = _resolve_device(device)
    payload = torch.load(path, map_location=resolved_device, weights_only=True)
    required = {
        "schema_version",
        "family",
        "architecture",
        "training_config",
        "model_state",
        "history",
        "best_epoch",
        "best_validation_loss",
        "normalization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("checkpoint schema mismatch")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema_version")
    family = _canonical_family(payload["family"])
    architecture = NeuralFieldConfig.from_mapping(payload["architecture"])
    config = TrainingConfig.from_mapping(payload["training_config"])
    model = ScaleConditionedNeuralField(architecture).to(resolved_device)
    model.load_state_dict(payload["model_state"], strict=True)
    setattr(model, "_lid_family", family)
    model.eval()

    history_raw = payload["history"]
    if not isinstance(history_raw, list) or not history_raw:
        raise ValueError("checkpoint history must be a non-empty list")
    history: list[EpochMetrics] = []
    for item in history_raw:
        if not isinstance(item, dict) or set(item) != {
            "epoch",
            "train_loss",
            "validation_loss",
            "learning_rate",
        }:
            raise ValueError("checkpoint history entry schema mismatch")
        history.append(EpochMetrics(**item))
    normalization = payload["normalization"]
    if not isinstance(normalization, dict) or set(normalization) != {
        "mean",
        "scale",
        "preprocessing",
        "sha256",
    }:
        raise ValueError("checkpoint normalization schema mismatch")
    mean = torch.as_tensor(normalization["mean"]).detach().cpu().float().contiguous()
    if mean.shape != (architecture.ambient_dim,) or not torch.isfinite(mean).all():
        raise ValueError("checkpoint normalization mean is invalid")
    scale = float(normalization["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("checkpoint normalization scale is invalid")
    preprocessing, preprocessing_sha256 = _preprocessing_identity(
        mean, scale, normalized=config.normalize
    )
    if (
        normalization["preprocessing"] != preprocessing
        or normalization["sha256"] != preprocessing_sha256
    ):
        raise ValueError("checkpoint preprocessing identity mismatch")
    best_epoch = int(payload["best_epoch"])
    best_validation_loss = float(payload["best_validation_loss"])
    if best_epoch not in {metric.epoch for metric in history}:
        raise ValueError("checkpoint best_epoch is absent from history")
    if not math.isfinite(best_validation_loss):
        raise ValueError("checkpoint best_validation_loss must be finite")
    return TrainingResult(
        family=family,
        model=model,
        config=config,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        checkpoint_path=path,
        checkpoint_sha256=_checkpoint_sha256(path),
        normalization_mean=mean,
        normalization_scale=scale,
        preprocessing=preprocessing,
        preprocessing_sha256=preprocessing_sha256,
    )


def _model_and_context(
    model_or_result: ScaleConditionedNeuralField | TrainingResult,
    family: str | None,
) -> tuple[ScaleConditionedNeuralField, Family, Tensor, float]:
    if isinstance(model_or_result, TrainingResult):
        canonical = model_or_result.family
        if family is not None and _canonical_family(family) != canonical:
            raise ValueError("family does not match TrainingResult")
        return (
            model_or_result.model,
            canonical,
            model_or_result.normalization_mean,
            model_or_result.normalization_scale,
        )
    model = model_or_result
    inferred = family or getattr(model, "_lid_family", None)
    if inferred is None:
        raise ValueError("family is required when passing a bare model")
    canonical = _canonical_family(inferred)
    mean = torch.zeros(model.config.ambient_dim, dtype=torch.float32)
    return model, canonical, mean, 1.0


def predict_primitives(
    model_or_result: ScaleConditionedNeuralField | TrainingResult,
    query: Any,
    scale: float,
    *,
    family: str | None = None,
    divergence_backend: Literal["exact", "hutchinson"] = "hutchinson",
    trace_probes: int = 16,
    trace_seed: int = 0,
    batch_size: int = 256,
) -> FieldPrediction:
    """Evaluate score/velocity and divergence at one model-space scale.

    Scale grids are defined after the checkpointed preprocessing transform,
    exactly like ``sigma_min``/``sigma_max`` used during training.
    """

    model, canonical_family, mean, normalization_scale = _model_and_context(
        model_or_result, family
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if canonical_family == "rectified_flow" and not 0 < scale < 1:
        raise ValueError("rectified-flow scale t must lie strictly between 0 and 1")
    if divergence_backend not in {"exact", "hutchinson"}:
        raise ValueError("divergence_backend must be exact or hutchinson")
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if divergence_backend == "hutchinson" and (
        isinstance(trace_probes, bool) or trace_probes <= 0
    ):
        raise ValueError("trace_probes must be positive for Hutchinson divergence")
    if isinstance(trace_seed, bool) or trace_seed < 0:
        raise ValueError("trace_seed must be a non-negative integer")

    query_cpu = _flat_finite_data(query, name="query")
    if query_cpu.shape[1] != model.config.ambient_dim:
        raise ValueError("query ambient dimension does not match model")
    normalized = (query_cpu - mean.reshape(1, -1)) / normalization_scale
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    model.eval()
    generator: torch.Generator | None = None
    if divergence_backend == "hutchinson":
        generator = torch.Generator(device=device)
        generator.manual_seed(trace_seed)

    fields: list[Tensor] = []
    divergences: list[Tensor] = []
    points: list[Tensor] = []
    model_scale = scale
    if model_scale <= 0:
        raise ValueError("normalized diffusion scale must be positive")
    for start in range(0, normalized.shape[0], batch_size):
        data_batch = normalized[start : start + batch_size].to(
            device=device, dtype=dtype
        )
        evaluation_point = (
            data_batch
            if canonical_family == "gaussian_diffusion"
            else model_scale * data_batch
        )
        condition = torch.full(
            (data_batch.shape[0],),
            model_scale,
            device=device,
            dtype=dtype,
        )
        if divergence_backend == "exact":
            raw_divergence = exact_divergence(
                model, evaluation_point, condition, create_graph=False
            )
        else:
            assert generator is not None
            raw_divergence = hutchinson_divergence(
                model,
                evaluation_point,
                condition,
                num_probes=trace_probes,
                seed=None,
                generator=generator,
                create_graph=False,
            )
        with torch.no_grad():
            raw_field = model(evaluation_point, condition)
        if canonical_family == "gaussian_diffusion":
            variance = model_scale**2
            field = (raw_field - evaluation_point) / variance
            divergence = (
                raw_divergence - model.config.ambient_dim
            ) / variance
        else:
            field = raw_field
            divergence = raw_divergence
        fields.append(field.detach().cpu())
        divergences.append(divergence.detach().cpu())
        points.append(evaluation_point.detach().cpu())
    return FieldPrediction(
        field=np.asarray(torch.cat(fields).numpy(), dtype=np.float64),
        divergence=np.asarray(torch.cat(divergences).numpy(), dtype=np.float64),
        evaluation_point=np.asarray(torch.cat(points).numpy(), dtype=np.float64),
        condition=float(model_scale),
    )


def predict_lid(
    model_or_result: ScaleConditionedNeuralField | TrainingResult,
    query: Any,
    scale: float,
    *,
    family: str | None = None,
    readout: Literal["full", "response"] = "full",
    divergence_backend: Literal["exact", "hutchinson"] = "hutchinson",
    trace_probes: int = 16,
    trace_seed: int = 0,
    batch_size: int = 256,
) -> npt.NDArray[np.float64]:
    """Return one LID estimate per query using the paper's native readout."""

    model, canonical_family, _, _ = _model_and_context(model_or_result, family)
    prediction = predict_primitives(
        model_or_result,
        query,
        scale,
        family=canonical_family,
        divergence_backend=divergence_backend,
        trace_probes=trace_probes,
        trace_seed=trace_seed,
        batch_size=batch_size,
    )
    ambient_dim = model.config.ambient_dim
    if canonical_family == "gaussian_diffusion":
        if readout != "full":
            raise ValueError("gaussian diffusion supports only the full readout")
        return np.asarray(
            diffusion_flipd(
                prediction.field,
                prediction.divergence,
                sigma=prediction.condition,
                ambient_dim=ambient_dim,
            ),
            dtype=np.float64,
        )
    if readout == "response":
        return np.asarray(
            rectified_flow_response(
                prediction.divergence, t=scale, ambient_dim=ambient_dim
            ),
            dtype=np.float64,
        )
    if readout != "full":
        raise ValueError("readout must be full or response")
    normalized_query = _flat_finite_data(query, name="query")
    if isinstance(model_or_result, TrainingResult):
        normalized_query = (
            normalized_query - model_or_result.normalization_mean.reshape(1, -1)
        ) / model_or_result.normalization_scale
    return np.asarray(
        rectified_flow_full(
            prediction.field,
            prediction.divergence,
            normalized_query.numpy(),
            t=scale,
            ambient_dim=ambient_dim,
        ),
        dtype=np.float64,
    )
