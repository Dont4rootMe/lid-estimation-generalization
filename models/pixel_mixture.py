"""Fit-only independent-pixel prior used as a denoising preconditioner.

Each pixel has a point mass at the known raw zero level and one Gaussian
fitted to its nonzero values. This is a frozen initialization, not a learned
LID rule: the spatial residual still minimizes the original native loss.
"""
from __future__ import annotations

import torch
from torch import nn


class PixelMixturePrior(nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.register_buffer('zero',torch.zeros(dimension))
        self.register_buffer('probability',torch.full((dimension,),.1))
        self.register_buffer('positive_mean',torch.ones(dimension))
        self.register_buffer('positive_variance',torch.ones(dimension))
        self.register_buffer('fitted',torch.tensor(False))

    @torch.no_grad()
    def fit(self,clean,zero):
        # Float64 sums; only optimizer-fit features enter these moments.
        dimension=clean.shape[1]
        counts=torch.zeros(dimension,dtype=torch.float64,device=clean.device)
        sums=torch.zeros_like(counts)
        squares=torch.zeros_like(counts)
        for batch in clean.split(4096):
            nonzero=batch != zero[None]
            masked=batch.double()*nonzero
            counts+=nonzero.sum(0)
            sums+=masked.sum(0)
            squares+=masked.square().sum(0)
        mean=sums/counts.clamp_min(1)
        variance=(squares/counts.clamp_min(1)-mean.square()).clamp_min(1e-4)
        for name,value in [('zero',zero),('probability',(counts/len(clean)).clamp(1e-6,1-1e-6)),
                           ('positive_mean',mean),('positive_variance',variance)]:
            getattr(self,name).copy_(value.to(getattr(self,name)))
        self.fitted.fill_(True)

    def posterior(self,channel,noise_ratio):
        if not bool(self.fitted):
            raise ValueError('pixel mixture prior must be fitted on optimizer-fit data')
        variance=noise_ratio.square()
        total=self.positive_variance+variance
        odds=(self.probability.log()-torch.log1p(-self.probability)
              +.5*(variance.log()-total.log())
              +(channel-self.zero).square()/(2*variance)
              -(channel-self.positive_mean).square()/(2*total))
        occupied=odds.sigmoid()
        gain=self.positive_variance/total
        positive=self.positive_mean+gain*(channel-self.positive_mean)
        mean=self.zero+occupied*(positive-self.zero)
        gate=occupied+(1-occupied)*variance/(1+variance)
        return mean,gate,occupied,gain

    def native_error(self,clean,noise,a,b,occupied,gain):
        # Stable (prior_mean-clean)/b, without a small-noise subtraction.
        lam=b/a
        total=self.positive_variance+lam.square()
        return ((1-occupied)*(self.zero-clean)/b
                +occupied*(gain*noise-(lam/total)*(clean-self.positive_mean))/a)
