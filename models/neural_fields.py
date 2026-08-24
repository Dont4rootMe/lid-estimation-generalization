"""Small, scale-conditioned neural vector fields and trace estimators.

The models in this module are deliberately data-shape agnostic: every sample
is flattened before it enters the network and the vector-field output is
restored to the input shape.  Consequently the same implementation handles
coordinate vectors and image-shaped observations without an architecture
change.

The divergence helpers assume a batch-separable vector field (as is the case
for :class:`ScaleConditionedNeuralField`).  They return one trace per sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Mapping

import torch
from torch import Tensor, nn


ConditionTransform = Literal["linear", "log"]


@dataclass(frozen=True)
class NeuralFieldConfig:
    """Serializable architecture definition for one neural vector field."""

    ambient_dim: int
    hidden_dim: int = 256
    depth: int = 4
    condition_dim: int = 64
    fourier_features: int = 32
    max_condition_frequency: float = 100.0
    dropout: float = 0.0
    condition_transform: ConditionTransform = "linear"
    zero_init_output: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.ambient_dim, bool) or self.ambient_dim <= 0:
            raise ValueError("ambient_dim must be a positive integer")
        if isinstance(self.hidden_dim, bool) or self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if isinstance(self.depth, bool) or self.depth <= 0:
            raise ValueError("depth must be a positive integer")
        if isinstance(self.condition_dim, bool) or self.condition_dim <= 0:
            raise ValueError("condition_dim must be a positive integer")
        if isinstance(self.fourier_features, bool) or self.fourier_features <= 0:
            raise ValueError("fourier_features must be a positive integer")
        if (
            not math.isfinite(self.max_condition_frequency)
            or self.max_condition_frequency < 1.0
        ):
            raise ValueError("max_condition_frequency must be finite and >= 1")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.condition_transform not in {"linear", "log"}:
            raise ValueError("condition_transform must be 'linear' or 'log'")
        if not isinstance(self.zero_init_output, bool):
            raise ValueError("zero_init_output must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NeuralFieldConfig":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown neural-field settings: {sorted(unknown)}")
        return cls(**dict(value))


class ScalarConditionEmbedding(nn.Module):
    """Deterministic Fourier embedding for time or positive noise scale."""

    def __init__(
        self,
        *,
        output_dim: int,
        fourier_features: int,
        max_frequency: float,
        transform: ConditionTransform,
    ) -> None:
        super().__init__()
        self.transform = transform
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(max_frequency), fourier_features)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        feature_dim = 1 + 2 * fourier_features
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, condition: Tensor) -> Tensor:
        if condition.ndim == 0:
            condition = condition.reshape(1, 1)
        elif condition.ndim == 1:
            condition = condition[:, None]
        elif condition.ndim != 2 or condition.shape[1] != 1:
            raise ValueError("condition must be scalar, (batch,), or (batch, 1)")
        if not torch.is_floating_point(condition):
            condition = condition.float()
        if not torch.isfinite(condition).all():
            raise ValueError("condition contains non-finite values")
        if self.transform == "log":
            if torch.any(condition <= 0):
                raise ValueError("log-conditioned scales must be strictly positive")
            condition = torch.log(condition)
        angles = condition * self.frequencies.to(
            device=condition.device, dtype=condition.dtype
        )[None, :]
        features = torch.cat(
            (condition, torch.sin(angles), torch.cos(angles)), dim=-1
        )
        return self.projection(features)


class ConditionedResidualBlock(nn.Module):
    """Pre-normalized residual MLP block with additive conditioning."""

    def __init__(self, hidden_dim: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.condition = nn.Linear(condition_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, state: Tensor, condition: Tensor) -> Tensor:
        update = self.hidden(self.norm(state)) + self.condition(condition)
        update = self.output(self.dropout(self.activation(update)))
        return (state + update) * (2.0**-0.5)


class ScaleConditionedNeuralField(nn.Module):
    """Residual MLP vector field conditioned on scalar time or noise scale."""

    def __init__(self, config: NeuralFieldConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_embedding = ScalarConditionEmbedding(
            output_dim=config.condition_dim,
            fourier_features=config.fourier_features,
            max_frequency=config.max_condition_frequency,
            transform=config.condition_transform,
        )
        self.input = nn.Linear(config.ambient_dim, config.hidden_dim)
        self.input_condition = nn.Linear(
            config.condition_dim, config.hidden_dim, bias=False
        )
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(
                config.hidden_dim, config.condition_dim, config.dropout
            )
            for _ in range(config.depth)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.ambient_dim),
        )
        if config.zero_init_output:
            output_layer = self.output[-1]
            assert isinstance(output_layer, nn.Linear)
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def forward(self, inputs: Tensor, condition: Tensor | float) -> Tensor:
        if inputs.ndim < 2:
            raise ValueError("inputs must have a batch dimension and feature dimensions")
        if not torch.is_floating_point(inputs):
            raise ValueError("inputs must have a floating-point dtype")
        if not torch.isfinite(inputs).all():
            raise ValueError("inputs contain non-finite values")
        batch_size = inputs.shape[0]
        flattened = inputs.reshape(batch_size, -1)
        if flattened.shape[1] != self.config.ambient_dim:
            raise ValueError(
                "flattened input dimension does not match ambient_dim: "
                f"{flattened.shape[1]} != {self.config.ambient_dim}"
            )
        condition_tensor = torch.as_tensor(
            condition, device=inputs.device, dtype=inputs.dtype
        )
        if condition_tensor.ndim == 0:
            condition_tensor = condition_tensor.expand(batch_size)
        elif condition_tensor.shape[0] == 1 and batch_size != 1:
            condition_tensor = condition_tensor.expand(batch_size, *condition_tensor.shape[1:])
        if condition_tensor.shape[0] != batch_size:
            raise ValueError("condition batch size does not match inputs")
        embedded = self.condition_embedding(condition_tensor)
        state = self.input(flattened) + self.input_condition(embedded)
        for block in self.blocks:
            state = block(state, embedded)
        return self.output(state).reshape_as(inputs)


def _differentiable_input(inputs: Tensor) -> Tensor:
    if not torch.is_floating_point(inputs):
        raise ValueError("divergence inputs must have a floating-point dtype")
    if inputs.ndim < 2 or inputs.shape[0] <= 0:
        raise ValueError("divergence inputs must have shape (batch, ...)")
    if inputs.requires_grad:
        return inputs
    return inputs.detach().requires_grad_(True)


def exact_divergence(
    field: nn.Module,
    inputs: Tensor,
    condition: Tensor | float,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Compute an exact per-sample Jacobian trace.

    Runtime scales linearly with the flattened ambient dimension, so this is
    intended as a reference implementation and for low-dimensional data.
    """

    with torch.enable_grad():
        differentiable = _differentiable_input(inputs)
        outputs = field(differentiable, condition)
        flat_inputs = differentiable.reshape(differentiable.shape[0], -1)
        flat_outputs = outputs.reshape(outputs.shape[0], -1)
        if flat_outputs.shape != flat_inputs.shape:
            raise ValueError("field output must have the same shape as its input")
        dimension = flat_inputs.shape[1]
        divergence = torch.zeros(
            flat_inputs.shape[0], device=inputs.device, dtype=inputs.dtype
        )
        for coordinate in range(dimension):
            gradient = torch.autograd.grad(
                flat_outputs[:, coordinate].sum(),
                differentiable,
                create_graph=create_graph,
                retain_graph=create_graph or coordinate + 1 < dimension,
                allow_unused=False,
            )[0]
            divergence = divergence + gradient.reshape(
                gradient.shape[0], -1
            )[:, coordinate]
        return divergence


def rademacher_probes_like(
    inputs: Tensor,
    *,
    num_probes: int,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Create deterministic Rademacher probes with shape ``(B, P, ...)``."""

    if isinstance(num_probes, bool) or num_probes <= 0:
        raise ValueError("num_probes must be a positive integer")
    if seed is not None and generator is not None:
        raise ValueError("pass either seed or generator, not both")
    if generator is None:
        if seed is None:
            raise ValueError("a seed or generator is required")
        if isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(seed)
    integer_probes = torch.randint(
        0,
        2,
        (inputs.shape[0], num_probes, *inputs.shape[1:]),
        generator=generator,
        device=inputs.device,
        dtype=torch.int64,
    )
    return integer_probes.to(dtype=inputs.dtype).mul_(2.0).sub_(1.0)


def hutchinson_divergence(
    field: nn.Module,
    inputs: Tensor,
    condition: Tensor | float,
    *,
    num_probes: int,
    seed: int | None = 0,
    generator: torch.Generator | None = None,
    probes: Tensor | None = None,
    create_graph: bool = False,
) -> Tensor:
    """Estimate a per-sample divergence using deterministic Rademacher probes."""

    if probes is not None and (generator is not None or seed not in {None, 0}):
        raise ValueError("explicit probes cannot be combined with a generator or seed")
    with torch.enable_grad():
        differentiable = _differentiable_input(inputs)
        outputs = field(differentiable, condition)
        if outputs.shape != differentiable.shape:
            raise ValueError("field output must have the same shape as its input")
        if probes is None:
            probes = rademacher_probes_like(
                differentiable,
                num_probes=num_probes,
                seed=seed if generator is None else None,
                generator=generator,
            )
        expected_shape = (
            differentiable.shape[0],
            num_probes,
            *differentiable.shape[1:],
        )
        if probes.shape != expected_shape:
            raise ValueError(
                f"probes must have shape {expected_shape}; got {tuple(probes.shape)}"
            )
        probes = probes.to(device=inputs.device, dtype=inputs.dtype)
        estimates: list[Tensor] = []
        for probe_index in range(num_probes):
            probe = probes[:, probe_index]
            vector_jacobian = torch.autograd.grad(
                outputs,
                differentiable,
                grad_outputs=probe,
                create_graph=create_graph,
                retain_graph=create_graph or probe_index + 1 < num_probes,
                allow_unused=False,
            )[0]
            estimates.append(
                (vector_jacobian * probe).reshape(inputs.shape[0], -1).sum(dim=1)
            )
        return torch.stack(estimates, dim=0).mean(dim=0)
