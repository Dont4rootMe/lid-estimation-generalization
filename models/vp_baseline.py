"""VP/FLIPD pilot primitives; deliberately separate from historical VE models.

The bottleneck topology and negative-noise objective follow the public FLIPD
MLP recipe. This is a reduced-width reconstruction, not its unpublished
benchmark training fork. See docs/VP_EXPERIMENT_HANDOFF.md for source revisions
and the proposed full protocol. Final LID formulas live in models.readouts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any, Literal

import torch
from torch import Tensor, nn

from models.neural_fields import exact_divergence, hutchinson_divergence


@dataclass(frozen=True)
class VPSchedule:
    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.beta_min, self.beta_max)):
            raise ValueError("VP beta endpoints must be finite")
        if not 0 < self.beta_min < self.beta_max:
            raise ValueError("VP requires 0 < beta_min < beta_max")

    def coefficients(self, time: Tensor) -> tuple[Tensor, Tensor]:
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("VP time must lie in [0, 1]")
        integral = (
            self.beta_min * time + 0.5 * (self.beta_max - self.beta_min) * time.square()
        )
        return torch.exp(-0.5 * integral), torch.sqrt(-torch.expm1(-integral))

    def time_for_lambda(self, noise_ratio: float) -> float:
        if not math.isfinite(noise_ratio) or noise_ratio <= 0:
            raise ValueError("lambda must be finite and positive")
        integral = math.log1p(noise_ratio**2)
        # Rationalized quadratic root avoids cancellation at small lambda.
        time = (
            2
            * integral
            / (
                self.beta_min
                + math.sqrt(
                    self.beta_min**2 + 2 * (self.beta_max - self.beta_min) * integral
                )
            )
        )
        if time > 1:
            raise ValueError("lambda lies beyond the VP training interval")
        return time

    def contract(self) -> dict:
        return {
            "family": "vp_diffusion_negative_noise",
            "schedule": asdict(self),
            "training_time": "uniform [0, 1)",
            "target": "negative standard normal noise",
            "loss": "batch mean of feature-summed squared error",
            "readout_query": "alpha(t) * normalized clean query",
            "scale_coordinate": "lambda = sigma(t) / alpha(t)",
        }


DEFAULT_VP_SCHEDULE = VPSchedule()


@dataclass(frozen=True)
class VPBottleneckConfig:
    """Serializable architecture definition for the reduced FLIPD MLP."""

    ambient_dim: int
    hidden_sizes: tuple[int, ...] = (512, 256, 128, 128, 64, 64)
    time_dim: int = 128
    condition_transform: Literal["linear", "log"] = "linear"

    def __post_init__(self) -> None:
        if isinstance(self.ambient_dim, bool) or self.ambient_dim < 1:
            raise ValueError("ambient_dim must be a positive integer")
        if len(self.hidden_sizes) < 2 or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain at least two positive integers")
        if (
            isinstance(self.time_dim, bool)
            or not isinstance(self.time_dim, int)
            or self.time_dim < 4
            or self.time_dim % 2
        ):
            raise ValueError("time_dim must be even and at least four")
        if self.condition_transform not in {"linear", "log"}:
            raise ValueError("condition_transform must be linear or log")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> VPBottleneckConfig:
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown VP architecture settings: {sorted(unknown)}")
        fields = dict(value)
        hidden_sizes = fields.get("hidden_sizes")
        if isinstance(hidden_sizes, list):
            fields["hidden_sizes"] = tuple(hidden_sizes)
        return cls(**fields)


class ConditionedBottleneckMLP(nn.Module):
    """Fully connected U-shaped field with concatenative skip connections.

    hidden_sizes describes the encoder, including the bottleneck; the decoder
    mirrors all preceding widths. Sinusoidal time features are concatenated
    with the input, as in the public FLIPD MLP implementation.
    """

    def __init__(
        self,
        ambient_dim: int,
        hidden_sizes=(512, 256, 128, 128, 64, 64),
        time_dim=128,
        *,
        condition_transform: Literal["linear", "log"] = "linear",
    ):
        super().__init__()
        self.config = VPBottleneckConfig(
            ambient_dim=ambient_dim,
            hidden_sizes=tuple(hidden_sizes),
            time_dim=time_dim,
            condition_transform=condition_transform,
        )
        self.ambient_dim = self.config.ambient_dim
        self.hidden_sizes = self.config.hidden_sizes
        self.time_dim = self.config.time_dim
        widths = [self.ambient_dim + self.time_dim, *self.hidden_sizes]
        self.encoder = nn.ModuleList(nn.Linear(a, b) for a, b in pairwise(widths))
        self.decoder = nn.ModuleList()
        previous = widths[-1]
        for index, skip_width in enumerate(reversed(widths[:-1])):
            output_width = ambient_dim if index == len(hidden_sizes) - 1 else skip_width
            self.decoder.append(nn.Linear(previous + skip_width, output_width))
            previous = output_width
        self.activation = nn.SiLU()

    def forward(self, inputs: Tensor, time: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.ambient_dim:
            raise ValueError("inputs must be (batch, ambient_dim)")
        condition = torch.as_tensor(time, device=inputs.device, dtype=inputs.dtype)
        if condition.ndim == 0:
            condition = condition.expand(inputs.shape[0])
        if condition.shape != (inputs.shape[0],):
            raise ValueError("condition must be scalar or (batch,)")
        if not torch.isfinite(condition).all():
            raise ValueError("condition must be finite")
        if self.config.condition_transform == "log":
            if torch.any(condition <= 0):
                raise ValueError("log-conditioned values must be positive")
            condition = torch.log(condition)
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=inputs.device, dtype=inputs.dtype)
            / (half - 1)
        )
        angles = condition[:, None] * frequencies[None, :]
        state = torch.cat((inputs, angles.sin(), angles.cos()), dim=1)
        skips = [state]
        for layer in self.encoder:
            state = self.activation(layer(state))
            skips.append(state)
        for index, (layer, skip) in enumerate(zip(self.decoder, reversed(skips[:-1]))):
            state = layer(torch.cat((skip, state), dim=1))
            if index + 1 < len(self.decoder):
                state = self.activation(state)
        return state


class VPBottleneckMLP(ConditionedBottleneckMLP):
    """VP specialization of the shared bottleneck with native-time input."""

    def __init__(
        self, ambient_dim: int, hidden_sizes=(512, 256, 128, 128, 64, 64), time_dim=128
    ) -> None:
        super().__init__(
            ambient_dim,
            hidden_sizes,
            time_dim,
            condition_transform="linear",
        )


def bottleneck_parameter_count(
    ambient_dim: int,
    hidden_sizes: tuple[int, ...],
    time_dim: int,
    *,
    condition_transform: Literal["linear", "log"] = "linear",
) -> int:
    """Return the exact parameter count without allocating real weights."""

    with torch.device("meta"):
        model = ConditionedBottleneckMLP(
            ambient_dim,
            hidden_sizes,
            time_dim,
            condition_transform=condition_transform,
        )
    return sum(parameter.numel() for parameter in model.parameters())


def vp_loss(
    model: nn.Module,
    clean: Tensor,
    time: Tensor,
    noise: Tensor,
    schedule: VPSchedule = DEFAULT_VP_SCHEDULE,
) -> Tensor:
    alpha, sigma = schedule.coefficients(time)
    noisy = alpha[:, None] * clean + sigma[:, None] * noise
    return (model(noisy, time) + noise).square().sum(dim=1).mean()


def denoise(
    model: nn.Module,
    noisy: Tensor,
    time: Tensor,
    schedule: VPSchedule = DEFAULT_VP_SCHEDULE,
) -> Tensor:
    alpha, sigma = schedule.coefficients(time)
    return (noisy + sigma[:, None] * model(noisy, time)) / alpha[:, None]


@dataclass(frozen=True)
class VPPrimitives:
    score: Tensor
    score_divergence: Tensor
    native_query: Tensor
    time: float
    alpha: float
    sigma: float


def vp_primitives(
    model: nn.Module,
    query: Tensor,
    noise_ratio: float,
    *,
    schedule: VPSchedule = DEFAULT_VP_SCHEDULE,
    probes=0,
    seed=0,
    generator: torch.Generator | None = None,
) -> VPPrimitives:
    """Export score and native-input divergence at y=alpha*x.

    No random corruption is added to the query. The trace is with respect to
    the native VP input y, not the original clean query x.
    """
    time = torch.full(
        (query.shape[0],),
        schedule.time_for_lambda(noise_ratio),
        device=query.device,
        dtype=query.dtype,
    )
    alpha, sigma = schedule.coefficients(time)
    native = alpha[:, None] * query
    if probes:
        divergence = hutchinson_divergence(
            model,
            native,
            time,
            num_probes=probes,
            seed=None if generator is not None else seed,
            generator=generator,
        )
    else:
        divergence = exact_divergence(model, native, time)
    with torch.no_grad():
        raw = model(native, time)
    return VPPrimitives(
        score=raw / sigma[:, None],
        score_divergence=divergence / sigma,
        native_query=native,
        time=float(time[0]),
        alpha=float(alpha[0]),
        sigma=float(sigma[0]),
    )
