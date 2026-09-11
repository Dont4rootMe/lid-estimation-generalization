"""Rao--Blackwellized targets for Gaussian-corruption training.

The bank contains optimizer-fit observations only. Conditioning the sampled
clean target on its noisy observation removes conditional label noise without
changing the expected squared-loss parameter gradient for that empirical prior.
Float64 logits avoid catastrophic cancellation at the smallest trained scale.
"""
import torch


class EmpiricalPosteriorTarget:
    def __init__(self, clean):
        self.clean=clean.detach().double().contiguous()
        self.half_norm=.5*self.clean.square().sum(1)

    def __deepcopy__(self,memo):
        # Immutable training data may be shared with the EMA copy. It is not
        # serialized in model state, and inference never needs this teacher.
        return self

    @torch.no_grad()
    def __call__(self, observations, scale):
        y=observations.detach().double()
        s=torch.as_tensor(scale,device=y.device,dtype=y.dtype).reshape(-1)
        if len(s)==1:s=s.expand(len(y))
        if s.shape!=(len(y),) or torch.any(s<0) or not torch.isfinite(s).all():
            raise ValueError('one finite nonnegative noise scale per query required')
        safe=s.clamp_min(torch.finfo(observations.dtype).eps)
        logits=(y@self.clean.T-self.half_norm[None,:])/safe[:,None].square()
        weights=torch.softmax(logits,dim=1)
        mean=(weights@self.clean).to(dtype=observations.dtype)
        return torch.where((s==0)[:,None],observations.detach(),mean)


def empirical_target_enabled(model):
    return model.training and getattr(model,'_lid_use_empirical_target',False)


def denoising_target(model,observations,scale,fallback):
    if not empirical_target_enabled(model):return fallback
    teacher=getattr(model,'_lid_empirical_teacher',None)
    if teacher is None:raise RuntimeError('empirical target requires the optimizer-fit bank')
    return teacher(observations,scale)
