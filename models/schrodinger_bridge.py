"""Canonical generalized Brownian Schrödinger-bridge training primitives.

The bridge used by the pilot is intentionally explicit.  Its terminal
marginal is the data law ``mu_T`` and its Schrödinger factors are
``F = Lebesgue`` and ``G = mu_T``.  Consequently its initial marginal is the
Gaussian smoothing ``mu_0 = h_{gamma T} * mu_T`` and, at time-to-go
``tau = T - t``, its marginal is ``h_{gamma tau} * mu_T``.

This construction remains meaningful for a singular terminal law as the
generalized mixture of Brownian bridges described in the paper.  It also
makes the trainable primitive unambiguous: an ``x_T`` denoiser determines the
forward drift by ``b_plus(r, tau) = (E[x_T | r] - r) / tau``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

CANONICAL_BRIDGE_CONSTRUCTION = "terminal-data-lebesgue-factor-v1"
CANONICAL_REFERENCE_PROCESS = "brownian-motion"
CANONICAL_INITIAL_MARGINAL = "gaussian-convolution-of-terminal-data"
CANONICAL_TERMINAL_MARGINAL = "dataset-terminal-law"
CANONICAL_FACTOR_F = "lebesgue-measure"
CANONICAL_FACTOR_G = "dataset-terminal-law"
CANONICAL_BRIDGE_CONDITIONING = "time-to-go-tau"


@dataclass(frozen=True)
class BrownianBridgeSpec:
    """Hydra-serializable identity of the canonical bridge path."""

    construction: str
    reference_process: str
    initial_marginal: str
    terminal_marginal: str
    factor_f: str
    factor_g: str
    conditioning: str
    diffusivity: float
    terminal_time: float

    def __post_init__(self) -> None:
        if self.construction != CANONICAL_BRIDGE_CONSTRUCTION:
            raise ValueError(
                f"bridge_construction must be exactly {CANONICAL_BRIDGE_CONSTRUCTION!r}"
            )
        exact_strings = {
            "bridge_reference_process": (
                self.reference_process,
                CANONICAL_REFERENCE_PROCESS,
            ),
            "bridge_initial_marginal": (
                self.initial_marginal,
                CANONICAL_INITIAL_MARGINAL,
            ),
            "bridge_terminal_marginal": (
                self.terminal_marginal,
                CANONICAL_TERMINAL_MARGINAL,
            ),
            "bridge_factor_f": (self.factor_f, CANONICAL_FACTOR_F),
            "bridge_factor_g": (self.factor_g, CANONICAL_FACTOR_G),
            "bridge_conditioning": (
                self.conditioning,
                CANONICAL_BRIDGE_CONDITIONING,
            ),
        }
        for name, (actual, expected) in exact_strings.items():
            if actual != expected:
                raise ValueError(f"{name} must be exactly {expected!r}")
        if not math.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise ValueError("bridge_diffusivity must be finite and positive")
        if not math.isfinite(self.terminal_time) or self.terminal_time <= 0:
            raise ValueError("bridge_terminal_time must be finite and positive")


def validate_time_to_go_bounds(
    *, minimum: float, maximum: float, spec: BrownianBridgeSpec
) -> None:
    """Validate a strict interior ``tau`` interval for bridge training."""

    if not (
        math.isfinite(minimum)
        and math.isfinite(maximum)
        and 0 < minimum < maximum <= spec.terminal_time
    ):
        raise ValueError(
            "bridge time-to-go bounds must satisfy "
            "0 < minimum < maximum <= terminal_time"
        )


def brownian_bridge_contract(
    spec: BrownianBridgeSpec, *, tau_min: float, tau_max: float
) -> dict[str, Any]:
    """Return the exact checkpoint/provenance identity of the bridge family."""

    validate_time_to_go_bounds(minimum=tau_min, maximum=tau_max, spec=spec)
    return {
        "schema_version": 1,
        "family": "brownian_schrodinger_bridge",
        "construction": spec.construction,
        "reference_process": spec.reference_process,
        "initial_marginal": spec.initial_marginal,
        "terminal_marginal": spec.terminal_marginal,
        "factor_f": spec.factor_f,
        "factor_g": spec.factor_g,
        "conditioning": spec.conditioning,
        "diffusivity": float(spec.diffusivity),
        "terminal_time": float(spec.terminal_time),
        "tau_min": float(tau_min),
        "tau_max": float(tau_max),
        "trainable_primitive": "terminal-denoiser-to-forward-drift",
        "readout": "brownian-sb-forward-full",
    }


def brownian_sb_terminal_denoising_loss(
    model: nn.Module,
    terminal: Tensor,
    *,
    tau_min: float,
    tau_max: float,
    spec: BrownianBridgeSpec,
    generator: torch.Generator,
) -> Tensor:
    """Weighted denoising loss for the canonical terminal bridge.

    ``tau`` is sampled log-uniformly so every order of magnitude in the
    endpoint region receives equal probability.  The weighting is the stable
    ``x_T`` parameterization of Brownian drift matching: with
    ``sigma**2 = gamma * tau`` it is identical to weighting the corresponding
    score error by ``sigma**2``.
    """

    validate_time_to_go_bounds(minimum=tau_min, maximum=tau_max, spec=spec)
    if terminal.ndim < 2 or terminal.shape[0] <= 0:
        raise ValueError("terminal must have shape (nonempty batch, ...)")
    if not torch.is_floating_point(terminal):
        raise ValueError("terminal must have a floating-point dtype")
    if not torch.isfinite(terminal).all():
        raise ValueError("terminal contains non-finite values")

    uniform = torch.rand(
        terminal.shape[0],
        device=terminal.device,
        dtype=terminal.dtype,
        generator=generator,
    )
    tau = torch.exp(
        math.log(tau_min) + uniform * (math.log(tau_max) - math.log(tau_min))
    )
    noise = torch.randn(
        terminal.shape,
        device=terminal.device,
        dtype=terminal.dtype,
        generator=generator,
    )
    sigma = torch.sqrt(spec.diffusivity * tau)
    sigma_broadcast = sigma.reshape(-1, *([1] * (terminal.ndim - 1)))
    bridge_state = terminal + sigma_broadcast * noise
    denoised_terminal = model(bridge_state, tau)
    if denoised_terminal.shape != terminal.shape:
        raise ValueError("bridge denoiser output must have the terminal shape")
    return ((denoised_terminal - terminal) / sigma_broadcast).square().flatten(1).mean()


def denoiser_to_forward_drift(
    denoised_terminal: Tensor,
    denoiser_divergence: Tensor,
    bridge_state: Tensor,
    *,
    tau: float,
    ambient_dim: int,
) -> tuple[Tensor, Tensor]:
    """Convert an ``x_T`` denoiser and its trace to ``b_plus`` and its trace."""

    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if isinstance(ambient_dim, bool) or ambient_dim <= 0:
        raise ValueError("ambient_dim must be a positive integer")
    if denoised_terminal.shape != bridge_state.shape:
        raise ValueError("denoised_terminal and bridge_state shapes must match")
    if denoised_terminal.ndim < 2 or denoised_terminal.shape[0] <= 0:
        raise ValueError("bridge fields must have shape (nonempty batch, ...)")
    flattened_dim = math.prod(bridge_state.shape[1:])
    if flattened_dim != ambient_dim:
        raise ValueError("bridge_state flattened dimension does not match ambient_dim")
    divergence = denoiser_divergence.reshape(-1)
    if divergence.shape != (bridge_state.shape[0],):
        raise ValueError("denoiser_divergence must contain one value per sample")
    if not (
        torch.isfinite(denoised_terminal).all()
        and torch.isfinite(bridge_state).all()
        and torch.isfinite(divergence).all()
    ):
        raise ValueError("bridge fields contain non-finite values")
    forward_drift = (denoised_terminal - bridge_state) / tau
    forward_drift_divergence = (divergence - ambient_dim) / tau
    return forward_drift, forward_drift_divergence
