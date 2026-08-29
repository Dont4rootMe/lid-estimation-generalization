"""Scale-conditioned normalizing flow for the paper's fixed-point interface.

The model is an explicit conditional RealNVP diffeomorphism.  At training
time it fits the coherent density path

``p_epsilon = Law(X_normalized + epsilon * standard_normal)``

with one shared set of parameters conditioned on the *known* smoothing scale.
Consequently ``log_prob(x, epsilon)`` is an exact change-of-variables density,
not a density proxy or a precomputed field bundle.  The LID readout is obtained
by differentiating this exact likelihood at a fixed observation:

``D + d / d log(epsilon) log p_epsilon(x)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from models.neural_fields import ScalarConditionEmbedding

NF_DENSITY_CONTRACT: Mapping[str, str | int] = {
    "schema_version": 1,
    "family": "scale_conditioned_normalizing_flow",
    "density_path": "law(x_normalized + epsilon * standard_normal)",
    "scale_coordinate": "log_epsilon",
    "likelihood": "exact_change_of_variables",
    "readout": "ambient_dim + d_log_epsilon log_p_epsilon_at_fixed_x",
    "scale_derivative": "autograd",
}


@dataclass(frozen=True)
class ConditionalFlowConfig:
    """Serializable architecture for an epsilon-conditioned RealNVP."""

    ambient_dim: int
    hidden_dim: int = 256
    num_coupling_layers: int = 6
    conditioner_depth: int = 2
    condition_dim: int = 64
    fourier_features: int = 32
    max_condition_frequency: float = 100.0
    dropout: float = 0.0
    log_scale_limit: float = 2.0

    def __post_init__(self) -> None:
        integer_positive = {
            "ambient_dim": self.ambient_dim,
            "hidden_dim": self.hidden_dim,
            "num_coupling_layers": self.num_coupling_layers,
            "conditioner_depth": self.conditioner_depth,
            "condition_dim": self.condition_dim,
            "fourier_features": self.fourier_features,
        }
        for name, value in integer_positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.ambient_dim < 2:
            raise ValueError("ambient_dim must be at least two for affine coupling")
        if (
            not math.isfinite(self.max_condition_frequency)
            or self.max_condition_frequency < 1.0
        ):
            raise ValueError("max_condition_frequency must be finite and >= 1")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not math.isfinite(self.log_scale_limit) or self.log_scale_limit <= 0.0:
            raise ValueError("log_scale_limit must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ConditionalFlowConfig:
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown conditional-flow settings: {sorted(unknown)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class FixedPointLikelihoodReadouts:
    """Likelihood-path diagnostics and LID readouts at one fixed point.

    The finite-difference and OLS bandwidths are deliberately separate.  A
    small finite-difference step audits the instantaneous autograd derivative,
    while the wider OLS window is a scientifically distinct smoothed readout.
    All likelihood values are evaluated at the unchanged observations.
    """

    epsilon: float
    finite_difference_log_step: float
    ols_log_step: float
    finite_difference_epsilons: Tensor
    finite_difference_log_likelihood: Tensor
    ols_epsilons: Tensor
    ols_log_likelihood: Tensor
    lid_autograd: Tensor
    lid_symmetric_fd: Tensor
    lid_ols3: Tensor
    lid_ols5: Tensor
    lid_ols9: Tensor


class _AffineCoupling(nn.Module):
    """One conditional affine coupling with an analytic inverse/Jacobian."""

    def __init__(
        self,
        *,
        ambient_dim: int,
        identity_indices: Tensor,
        transform_indices: Tensor,
        condition_dim: int,
        hidden_dim: int,
        conditioner_depth: int,
        dropout: float,
        log_scale_limit: float,
    ) -> None:
        super().__init__()
        if identity_indices.numel() == 0 or transform_indices.numel() == 0:
            raise ValueError("each coupling partition must be non-empty")
        if identity_indices.numel() + transform_indices.numel() != ambient_dim:
            raise ValueError("coupling partitions must cover the ambient coordinates")
        self.register_buffer(
            "identity_indices", identity_indices.to(dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "transform_indices", transform_indices.to(dtype=torch.long), persistent=True
        )
        self.log_scale_limit = float(log_scale_limit)
        layers: list[nn.Module] = []
        input_dim = int(identity_indices.numel()) + condition_dim
        for layer_index in range(conditioner_depth):
            layers.append(
                nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim)
            )
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        output = nn.Linear(hidden_dim, 2 * int(transform_indices.numel()))
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.conditioner = nn.Sequential(*layers)

    def _affine_parameters(
        self, fixed: Tensor, condition: Tensor
    ) -> tuple[Tensor, Tensor]:
        raw_scale, translation = self.conditioner(
            torch.cat((fixed, condition), dim=-1)
        ).chunk(2, dim=-1)
        log_scale = self.log_scale_limit * torch.tanh(raw_scale)
        return log_scale, translation

    def forward(self, inputs: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        fixed = inputs.index_select(1, self.identity_indices)
        transformed = inputs.index_select(1, self.transform_indices)
        log_scale, translation = self._affine_parameters(fixed, condition)
        outputs = inputs.clone()
        outputs[:, self.transform_indices] = (
            transformed * torch.exp(log_scale) + translation
        )
        return outputs, log_scale.sum(dim=-1)

    def inverse(self, outputs: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        fixed = outputs.index_select(1, self.identity_indices)
        transformed = outputs.index_select(1, self.transform_indices)
        log_scale, translation = self._affine_parameters(fixed, condition)
        inputs = outputs.clone()
        inputs[:, self.transform_indices] = (transformed - translation) * torch.exp(
            -log_scale
        )
        return inputs, -log_scale.sum(dim=-1)


class ScaleConditionedRealNVP(nn.Module):
    """A coherent conditional family of regular, exactly normalized densities."""

    def __init__(self, config: ConditionalFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_embedding = ScalarConditionEmbedding(
            output_dim=config.condition_dim,
            fourier_features=config.fourier_features,
            max_frequency=config.max_condition_frequency,
            transform="log",
        )
        even = torch.arange(0, config.ambient_dim, 2, dtype=torch.long)
        odd = torch.arange(1, config.ambient_dim, 2, dtype=torch.long)
        couplings: list[_AffineCoupling] = []
        for layer_index in range(config.num_coupling_layers):
            identity, transform = (even, odd) if layer_index % 2 == 0 else (odd, even)
            couplings.append(
                _AffineCoupling(
                    ambient_dim=config.ambient_dim,
                    identity_indices=identity,
                    transform_indices=transform,
                    condition_dim=config.condition_dim,
                    hidden_dim=config.hidden_dim,
                    conditioner_depth=config.conditioner_depth,
                    dropout=config.dropout,
                    log_scale_limit=config.log_scale_limit,
                )
            )
        self.couplings = nn.ModuleList(couplings)

    def _flatten(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 2 or inputs.shape[0] <= 0:
            raise ValueError("inputs must have shape (nonempty batch, ...)")
        if not torch.is_floating_point(inputs):
            raise ValueError("inputs must have a floating-point dtype")
        if not torch.isfinite(inputs).all():
            raise ValueError("inputs contain non-finite values")
        flattened = inputs.reshape(inputs.shape[0], -1)
        if flattened.shape[1] != self.config.ambient_dim:
            raise ValueError(
                "flattened input dimension does not match ambient_dim: "
                f"{flattened.shape[1]} != {self.config.ambient_dim}"
            )
        return flattened

    def _condition(self, epsilon: Tensor | float, *, reference: Tensor) -> Tensor:
        value = torch.as_tensor(epsilon, device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.expand(reference.shape[0])
        elif value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        elif value.ndim != 1:
            raise ValueError("epsilon must be scalar, (batch,), or (batch, 1)")
        if value.shape[0] == 1 and reference.shape[0] != 1:
            value = value.expand(reference.shape[0])
        if value.shape != (reference.shape[0],):
            raise ValueError("epsilon batch size does not match inputs")
        if not torch.isfinite(value).all() or torch.any(value <= 0):
            raise ValueError("epsilon must contain finite positive values")
        return self.condition_embedding(value)

    def encode(
        self, observations: Tensor, epsilon: Tensor | float
    ) -> tuple[Tensor, Tensor]:
        """Map observations to the base and return ``log|dz/dx|`` exactly."""

        state = self._flatten(observations)
        condition = self._condition(epsilon, reference=state)
        log_abs_det = torch.zeros(
            state.shape[0], device=state.device, dtype=state.dtype
        )
        for coupling in reversed(self.couplings):
            state, contribution = coupling.inverse(state, condition)
            log_abs_det = log_abs_det + contribution
        return state, log_abs_det

    def decode(self, latent: Tensor, epsilon: Tensor | float) -> tuple[Tensor, Tensor]:
        """Map base samples to observations and return ``log|dx/dz|`` exactly."""

        state = self._flatten(latent)
        condition = self._condition(epsilon, reference=state)
        log_abs_det = torch.zeros(
            state.shape[0], device=state.device, dtype=state.dtype
        )
        for coupling in self.couplings:
            state, contribution = coupling(state, condition)
            log_abs_det = log_abs_det + contribution
        return state, log_abs_det

    def log_prob(self, observations: Tensor, epsilon: Tensor | float) -> Tensor:
        """Evaluate the exactly normalized conditional log-density."""

        latent, inverse_log_abs_det = self.encode(observations, epsilon)
        base_log_prob = -0.5 * (latent.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        return base_log_prob + inverse_log_abs_det


def fixed_point_lid(
    model: ScaleConditionedRealNVP,
    observations: Tensor,
    epsilon: float,
) -> Tensor:
    """Evaluate ``D + partial_log(epsilon) log p_epsilon(x)`` at fixed ``x``."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    with torch.enable_grad():
        fixed = observations.detach()
        log_epsilon = torch.full(
            (fixed.shape[0],),
            math.log(epsilon),
            device=fixed.device,
            dtype=fixed.dtype,
            requires_grad=True,
        )
        likelihood = model.log_prob(fixed, torch.exp(log_epsilon))
        derivative = torch.autograd.grad(
            likelihood.sum(), log_epsilon, create_graph=False, retain_graph=False
        )[0]
        return derivative + model.config.ambient_dim


def _positive_finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _offset_epsilon_grid(
    observations: Tensor,
    *,
    epsilon: float,
    log_step: float,
    offsets: Sequence[int],
) -> Tensor:
    center = _positive_finite_scalar(epsilon, name="epsilon")
    step = _positive_finite_scalar(log_step, name="log_step")
    if not offsets:
        raise ValueError("offsets must be nonempty")
    if any(
        isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets
    ):
        raise TypeError("offsets must contain integers")
    requested_log_epsilon = torch.tensor(
        [math.log(center) + int(offset) * step for offset in offsets],
        device=observations.device,
        dtype=observations.dtype,
    )
    epsilon_grid = torch.exp(requested_log_epsilon)
    if not torch.isfinite(epsilon_grid).all() or torch.any(epsilon_grid <= 0):
        raise ValueError("epsilon grid is not representable in the observation dtype")
    if epsilon_grid.numel() > 1 and not torch.all(epsilon_grid[1:] > epsilon_grid[:-1]):
        raise ValueError(
            "log_step is too small to produce a strictly increasing epsilon grid"
        )
    return epsilon_grid


def fixed_point_log_likelihood_curve(
    model: ScaleConditionedRealNVP,
    observations: Tensor,
    epsilons: Tensor | Sequence[float],
) -> Tensor:
    """Evaluate ``log p_epsilon(x)`` on a declared grid at unchanged ``x``.

    The scale loop avoids materializing a ``batch x scales x ambient_dim``
    tensor, which is prohibitive for image-space flows.  The returned array has
    shape ``(batch, scales)`` and follows the model/observation dtype.
    """

    if observations.ndim < 2 or observations.shape[0] <= 0:
        raise ValueError("observations must have shape (nonempty batch, ...)")
    if not torch.is_floating_point(observations):
        raise ValueError("observations must have a floating-point dtype")
    fixed = observations.detach()
    epsilon_grid = torch.as_tensor(
        epsilons,
        device=fixed.device,
        dtype=fixed.dtype,
    ).detach()
    if epsilon_grid.ndim != 1 or epsilon_grid.numel() <= 0:
        raise ValueError("epsilons must be a nonempty rank-one grid")
    if not torch.isfinite(epsilon_grid).all() or torch.any(epsilon_grid <= 0):
        raise ValueError("epsilons must contain finite positive values")
    if epsilon_grid.numel() > 1 and not torch.all(epsilon_grid[1:] > epsilon_grid[:-1]):
        raise ValueError("epsilons must be strictly increasing")
    with torch.no_grad():
        columns = [model.log_prob(fixed, value) for value in epsilon_grid]
        result = torch.stack(columns, dim=1)
    if result.shape != (fixed.shape[0], epsilon_grid.numel()):
        raise RuntimeError("normalizing-flow likelihood curve has the wrong shape")
    if not torch.isfinite(result).all():
        raise FloatingPointError("normalizing-flow likelihood curve is non-finite")
    return result


def _local_ols_lid_from_curve(
    model: ScaleConditionedRealNVP,
    log_likelihood: Tensor,
    epsilons: Tensor,
) -> Tensor:
    if log_likelihood.ndim != 2:
        raise ValueError("log_likelihood must have shape (batch, scales)")
    if epsilons.ndim != 1 or log_likelihood.shape[1] != epsilons.numel():
        raise ValueError("likelihood curve and epsilon grid shapes do not match")
    if epsilons.numel() < 3 or epsilons.numel() % 2 == 0:
        raise ValueError(
            "local OLS requires an odd window containing at least 3 points"
        )
    if not torch.isfinite(log_likelihood).all():
        raise FloatingPointError("log_likelihood contains non-finite values")
    log_grid = torch.log(epsilons.detach().to(dtype=torch.float64))
    centered_grid = log_grid - log_grid.mean()
    denominator = centered_grid.square().sum()
    if not torch.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("epsilon grid is degenerate in log space")
    values = log_likelihood.detach().to(dtype=torch.float64)
    center = values[:, values.shape[1] // 2 : values.shape[1] // 2 + 1]
    slope = torch.matmul(values - center, centered_grid) / denominator
    return slope + float(model.config.ambient_dim)


def fixed_point_lid_symmetric_finite_difference(
    model: ScaleConditionedRealNVP,
    observations: Tensor,
    epsilon: float,
    *,
    log_step: float,
) -> Tensor:
    """Estimate the fixed-point log-scale derivative by a centered difference."""

    epsilon_grid = _offset_epsilon_grid(
        observations,
        epsilon=epsilon,
        log_step=log_step,
        offsets=(-1, 1),
    )
    likelihood = fixed_point_log_likelihood_curve(model, observations, epsilon_grid)
    log_grid = torch.log(epsilon_grid.to(dtype=torch.float64))
    slope = (
        likelihood[:, 1].to(dtype=torch.float64)
        - likelihood[:, 0].to(dtype=torch.float64)
    ) / (log_grid[1] - log_grid[0])
    return slope + float(model.config.ambient_dim)


def fixed_point_lid_symmetric_fd(
    model: ScaleConditionedRealNVP,
    observations: Tensor,
    epsilon: float,
    *,
    log_step: float,
) -> Tensor:
    """Short alias for :func:`fixed_point_lid_symmetric_finite_difference`."""

    return fixed_point_lid_symmetric_finite_difference(
        model,
        observations,
        epsilon,
        log_step=log_step,
    )


def fixed_point_lid_local_ols(
    model: ScaleConditionedRealNVP,
    observations: Tensor,
    epsilon: float,
    *,
    log_step: float,
    window_size: int,
) -> Tensor:
    """Estimate the local likelihood slope by OLS in ``log(epsilon)``."""

    if isinstance(window_size, bool) or window_size not in {3, 5, 9}:
        raise ValueError("window_size must be one of 3, 5, or 9")
    radius = window_size // 2
    epsilon_grid = _offset_epsilon_grid(
        observations,
        epsilon=epsilon,
        log_step=log_step,
        offsets=tuple(range(-radius, radius + 1)),
    )
    likelihood = fixed_point_log_likelihood_curve(model, observations, epsilon_grid)
    return _local_ols_lid_from_curve(model, likelihood, epsilon_grid)


def fixed_point_likelihood_readouts(
    model: ScaleConditionedRealNVP,
    observations: Tensor,
    epsilon: float,
    *,
    finite_difference_log_step: float,
    ols_log_step: float,
) -> FixedPointLikelihoodReadouts:
    """Evaluate all NF ablation readouts while sharing likelihood-path work."""

    center = _positive_finite_scalar(epsilon, name="epsilon")
    fd_step = _positive_finite_scalar(
        finite_difference_log_step,
        name="finite_difference_log_step",
    )
    regression_step = _positive_finite_scalar(ols_log_step, name="ols_log_step")
    finite_difference_epsilons = _offset_epsilon_grid(
        observations,
        epsilon=center,
        log_step=fd_step,
        offsets=(-1, 1),
    )
    finite_difference_log_likelihood = fixed_point_log_likelihood_curve(
        model,
        observations,
        finite_difference_epsilons,
    )
    finite_difference_log_grid = torch.log(
        finite_difference_epsilons.to(dtype=torch.float64)
    )
    finite_difference_slope = (
        finite_difference_log_likelihood[:, 1].to(dtype=torch.float64)
        - finite_difference_log_likelihood[:, 0].to(dtype=torch.float64)
    ) / (finite_difference_log_grid[1] - finite_difference_log_grid[0])

    ols_epsilons = _offset_epsilon_grid(
        observations,
        epsilon=center,
        log_step=regression_step,
        offsets=tuple(range(-4, 5)),
    )
    ols_log_likelihood = fixed_point_log_likelihood_curve(
        model,
        observations,
        ols_epsilons,
    )
    return FixedPointLikelihoodReadouts(
        epsilon=center,
        finite_difference_log_step=fd_step,
        ols_log_step=regression_step,
        finite_difference_epsilons=finite_difference_epsilons.detach(),
        finite_difference_log_likelihood=finite_difference_log_likelihood.detach(),
        ols_epsilons=ols_epsilons.detach(),
        ols_log_likelihood=ols_log_likelihood.detach(),
        lid_autograd=fixed_point_lid(model, observations, center)
        .detach()
        .to(dtype=torch.float64),
        lid_symmetric_fd=(
            finite_difference_slope + float(model.config.ambient_dim)
        ).detach(),
        lid_ols3=_local_ols_lid_from_curve(
            model,
            ols_log_likelihood[:, 3:6],
            ols_epsilons[3:6],
        ).detach(),
        lid_ols5=_local_ols_lid_from_curve(
            model,
            ols_log_likelihood[:, 2:7],
            ols_epsilons[2:7],
        ).detach(),
        lid_ols9=_local_ols_lid_from_curve(
            model,
            ols_log_likelihood,
            ols_epsilons,
        ).detach(),
    )


def conditional_smoothed_nll(
    model: ScaleConditionedRealNVP,
    clean: Tensor,
    *,
    epsilon_min: float,
    epsilon_max: float,
    generator: torch.Generator,
) -> Tensor:
    """Monte-Carlo MLE objective for the coherent Gaussian-smoothed path.

    The loss is normalized by ambient dimension for optimizer conditioning;
    ``model.log_prob`` itself always returns the full exact log-density.
    Equal bounds define a fixed-epsilon density and intentionally consume no
    uniform random draw before sampling its Gaussian perturbation.
    """

    if not (
        math.isfinite(epsilon_min)
        and math.isfinite(epsilon_max)
        and 0.0 < epsilon_min <= epsilon_max
    ):
        raise ValueError("epsilon bounds must satisfy 0 < min <= max")
    if epsilon_min == epsilon_max:
        epsilon = torch.full(
            (clean.shape[0],),
            float(epsilon_min),
            device=clean.device,
            dtype=clean.dtype,
        )
    else:
        uniform = torch.rand(
            clean.shape[0],
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        epsilon = torch.exp(
            math.log(epsilon_min)
            + uniform * (math.log(epsilon_max) - math.log(epsilon_min))
        )
    noise = torch.randn(
        clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    epsilon_broadcast = epsilon.reshape(-1, *([1] * (clean.ndim - 1)))
    noisy = clean + epsilon_broadcast * noise
    return -model.log_prob(noisy, epsilon).mean() / model.config.ambient_dim
