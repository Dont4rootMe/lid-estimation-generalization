"""Spatial residual field with native VE/RF Gaussian parameterizations.

The image remains in NHWC coordinates at the public interface. Convolutional
hidden skips have no fixed low-rank output span and no dense raw-input skip.
There is no activation normalization across pixels or channels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.gaussian_fields import NativeGaussianBottleneck
from models.pixel_mixture import PixelMixturePrior


@dataclass(frozen=True)
class ImageFieldConfig:
    ambient_dim: int
    shape: tuple[int, int, int] = (32,32,3)
    width: int = 32
    training_bf16: bool = False
    prior: str = 'unit_gaussian'
    tail_order: int = 1

    def __post_init__(self):
        if len(self.shape) != 3 or any(not isinstance(v,int) or isinstance(v,bool) or v <= 0 for v in self.shape):
            raise ValueError('image shape must contain three positive integers')
        if math.prod(self.shape) != self.ambient_dim or any(v % 4 for v in self.shape[:2]):
            raise ValueError('image shape must match ambient dimension and spatial sizes must divide by four')
        if not isinstance(self.width,int) or isinstance(self.width,bool) or self.width < 4:
            raise ValueError('image width must be an integer of at least four')
        if not isinstance(self.training_bf16,bool):
            raise ValueError('training_bf16 must be boolean')
        if self.prior not in {'unit_gaussian','pixel_spike_gaussian_v1'}:
            raise ValueError('unknown image prior')
        if self.tail_order not in {1,2} or (self.tail_order==2 and self.prior!='unit_gaussian'):
            raise ValueError('tail order two requires the unit Gaussian prior')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls,value):
        return cls(**{**dict(value),'shape':tuple(value['shape'])})


class ImageBlock(nn.Module):
    def __init__(self,incoming,outgoing):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming,outgoing,3,padding=1)
        self.condition = nn.Linear(128,outgoing)
        self.conv2 = nn.Conv2d(outgoing,outgoing,3,padding=1)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming,outgoing,1)

    def forward(self,value,condition):
        hidden = self.conv1(F.silu(value)) + self.condition(condition)[:,:,None,None]
        hidden = self.conv2(F.silu(hidden))
        return (hidden+self.skip(value))/math.sqrt(2)


class ImageUNet(nn.Module):
    def __init__(self,channels,width,output_channels=None):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        self.input = nn.Conv2d(channels,width,3,padding=1)
        self.high = ImageBlock(width,width)
        self.middle = ImageBlock(width,2*width)
        self.low1 = ImageBlock(2*width,4*width)
        self.low2 = ImageBlock(4*width,4*width)
        self.up_middle = ImageBlock(6*width,2*width)
        self.up_high = ImageBlock(3*width,width)
        self.output = nn.Conv2d(width,channels if output_channels is None else output_channels,3,padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,inputs,log_lambda):
        frequencies = torch.exp(-math.log(10000)*torch.arange(64,device=inputs.device,dtype=inputs.dtype)/63)
        angles = log_lambda[:,None]*frequencies[None]
        condition = self.time(torch.cat((angles.sin(),angles.cos()),1))
        high = self.high(self.input(inputs),condition)
        middle = self.middle(F.avg_pool2d(high,2),condition)
        low = self.low2(self.low1(F.avg_pool2d(middle,2),condition),condition)
        middle_out = self.up_middle(torch.cat((F.interpolate(low,size=middle.shape[-2:],mode='nearest'),middle),1),condition)
        high_out = self.up_high(torch.cat((F.interpolate(middle_out,size=high.shape[-2:],mode='nearest'),high),1),condition)
        return self.output(F.silu(high_out))


class NativeGaussianImageField(nn.Module):
    coefficients = NativeGaussianBottleneck.coefficients

    def __init__(self,config: ImageFieldConfig,*,native_family: str):
        super().__init__()
        if native_family not in {'gaussian_diffusion','rectified_flow'}:
            raise ValueError('native Gaussian image field supports VE and rectified flow')
        self.config,self.native_family = config,native_family
        self.field = ImageUNet(config.shape[-1],config.width)
        if config.prior=='pixel_spike_gaussian_v1':
            self.pixel_prior=PixelMixturePrior(config.ambient_dim)

    def forward(self,inputs,condition):
        if self.config.prior=='unit_gaussian' and self.config.tail_order==1:
            return NativeGaussianBottleneck.forward(self,inputs,condition)
        if self.config.tail_order==2:
            a,b,inverse=self.coefficients(inputs,condition)
            residual=self.residual(inputs*inverse,(b/a).flatten())
            if self.native_family=='gaussian_diffusion':
                return a*inverse.square()*inputs+a*b*inverse.square()*residual
            return (a-b)*inverse.square()*inputs+a*inverse.square()*residual
        a,b,inverse=self.coefficients(inputs,condition)
        lam=b/a
        mean,gate,_,_=self.pixel_prior.posterior(inputs/a,lam)
        residual=self.residual(inputs*inverse,lam.flatten())
        if self.native_family=='gaussian_diffusion':
            return mean+b*inverse*gate*residual
        return (mean-inputs)/b+inverse*gate*residual

    def residual_loss(self,clean,condition,noise):
        if self.config.prior=='unit_gaussian' and self.config.tail_order==1:
            return NativeGaussianBottleneck.residual_loss(self,clean,condition,noise)
        if self.config.tail_order==2:
            a,b,inverse=self.coefficients(clean,condition)
            residual=self.residual((a*clean+b*noise)*inverse,(b/a).flatten())
            return ((a*residual-b*clean+a*noise)*inverse.square()).square().mean()
        a,b,inverse=self.coefficients(clean,condition)
        native=a*clean+b*noise
        lam=b/a
        _,gate,occupied,gain=self.pixel_prior.posterior(native/a,lam)
        residual=self.residual(native*inverse,lam.flatten())
        error=self.pixel_prior.native_error(clean,noise,a,b,occupied,gain)
        return (error+inverse*gate*residual).square().mean()

    def residual(self,inputs,noise_ratio):
        if inputs.ndim != 2 or inputs.shape[1] != self.config.ambient_dim:
            raise ValueError('image field inputs must be flat NHWC')
        spatial = inputs.reshape(len(inputs),*self.config.shape).permute(0,3,1,2)
        mixed = self.config.training_bf16 and self.training and inputs.is_cuda and inputs.dtype == torch.float32
        with torch.autocast(device_type=inputs.device.type,dtype=torch.bfloat16,enabled=mixed):
            residual = self.field(spatial,noise_ratio.log())
        return residual.to(inputs.dtype).permute(0,2,3,1).reshape_as(inputs)
