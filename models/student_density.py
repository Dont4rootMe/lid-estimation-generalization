"""Full density LID for a radial Student kernel in per-coordinate RMS units."""
import math

import torch


def density_dilation(model, query, scale, response, df):
    """Return N + d_log(lambda) log rho from a posterior and its response.

    For the exact posterior b(y, lambda), put e=y-b, a=(nu-2)*lambda**2,
    s=||e||**2/a and R=div_y b. The Student continuity identities give
    F=(R+(N+nu)*s+e.dot(d_log(lambda)b)/a)/(1+s).
    The derivative holds canonical y fixed. This is NOT the Gaussian
    squared-displacement correction. Batch rows must be independent.
    """
    if not math.isfinite(df) or df <= 2:
        raise ValueError('finite-variance Student scale requires df > 2')
    if query.ndim != 2 or response.shape != (len(query),):
        raise ValueError('expected flat queries and one response per query')
    scale = torch.as_tensor(scale, device=query.device, dtype=query.dtype)
    if scale.ndim == 0:
        scale = scale.expand(len(query))
    if scale.shape != (len(query),) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('expected one positive finite RMS scale per query')
    with torch.enable_grad():
        log_scale = scale.log().detach().clone().requires_grad_(True)
        posterior = model(query, log_scale.exp())
        displacement = query - posterior
        # Detach e: only the posterior, not its multiplier, is differentiated.
        contraction = torch.autograd.grad(
            (displacement.detach() * posterior).sum(), log_scale, allow_unused=True)[0]
        if contraction is None:
            contraction = torch.zeros_like(log_scale)
        radius_squared = (df - 2) * log_scale.exp().square()
        relative_displacement = displacement.square().sum(1) / radius_squared
        full = (response + (query.shape[1] + df) * relative_displacement
                + contraction / radius_squared) / (1 + relative_displacement)
    return full.detach(), posterior.detach(), contraction.detach()
