"""Native radial-Student density dilation from a learned posterior field.

This is a supplementary readout; it does not change the frozen response
campaign, checkpoint selection, training source, or Gaussian readout API.
"""
import torch


def density_dilation(model, query, scale, response, df):
    """Return F=N+d_log(lambda) log rho, if model is the exact posterior.

    Let e=y-b, a=(nu-2)*lambda**2, and s=||e||**2/a. Then
    F=(R+(N+nu)*s+e.dot(d_log(lambda)b)/a)/(1+s), R=div_y b.
    The scale derivative is at fixed canonical y. Each batch row must be
    independent (as in the routed models, which use no batch normalization).
    """
    if df <= 2:
        raise ValueError('finite-variance Student scale requires df > 2')
    with torch.enable_grad():
        log_scale = torch.as_tensor(scale, device=query.device, dtype=query.dtype).log()
        if log_scale.ndim == 0:
            log_scale = log_scale.expand(len(query))
        log_scale = log_scale.detach().clone().requires_grad_(True)
        posterior = model(query, log_scale.exp())
        displacement = query - posterior
        # Detaching e prevents differentiating the multiplier instead of b.
        contraction = torch.autograd.grad(
            (displacement.detach() * posterior).sum(), log_scale)[0]
        radius_squared = (df - 2) * log_scale.exp().square()
        relative_displacement = displacement.square().sum(1) / radius_squared
        full = (response + (query.shape[1] + df) * relative_displacement
                + contraction / radius_squared) / (1 + relative_displacement)
    return full.detach(), posterior.detach(), contraction.detach()
