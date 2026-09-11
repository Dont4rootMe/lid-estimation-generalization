"""A shared residual backbone with noise-resolved spatial Fourier features.

The fixed frequency bank is the same for every data set. Gaussian attenuation
turns off frequencies finer than the current noise, without LID information.
"""
import math
import torch
from torch import nn


class ConditionedResidualBlock(nn.Module):
    def __init__(self,width,condition_dim):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.modulation=nn.Linear(condition_dim,2*width)
        self.first=nn.Linear(width,width);self.second=nn.Linear(width,width)
        nn.init.zeros_(self.second.weight);nn.init.zeros_(self.second.bias)

    def forward(self,x,condition):
        gain,bias=self.modulation(condition).chunk(2,1)
        h=self.norm(x)*(1+.1*gain)+.1*bias
        h=self.second(torch.nn.functional.silu(self.first(torch.nn.functional.silu(h))))
        return x+h


class SpectralResidualCore(nn.Module):
    def __init__(self,dimension,width=512,blocks=4,condition_dim=128):
        super().__init__()
        self.register_buffer('spatial_frequencies',math.pi*2.**torch.arange(10))
        self.register_buffer('condition_frequencies',torch.exp(-math.log(10000)*torch.arange(condition_dim//2)/(condition_dim//2-1)))
        self.condition=nn.Sequential(nn.Linear(condition_dim,condition_dim),nn.SiLU(),nn.Linear(condition_dim,condition_dim))
        self.input=nn.Linear(dimension*21,width)
        self.blocks=nn.ModuleList(ConditionedResidualBlock(width,condition_dim) for _ in range(blocks))
        self.output=nn.Linear(width,dimension)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x,log_scale,noise_level):
        angles=x[:,:,None]*self.spatial_frequencies
        attenuation=torch.exp(-.5*(noise_level[:,:,None]*self.spatial_frequencies).square())
        features=torch.cat((x,(angles.sin()*attenuation).flatten(1),(angles.cos()*attenuation).flatten(1)),1)
        time=log_scale[:,None]*self.condition_frequencies
        condition=self.condition(torch.cat((time.sin(),time.cos()),1))
        h=self.input(features)
        for block in self.blocks:h=block(h,condition)
        return self.output(torch.nn.functional.silu(h))
