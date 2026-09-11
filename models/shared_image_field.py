"""The same spatial posterior core for every native vector-field interface.

Native targets and samplers remain in models.training. This module changes
only the neural parameterization; it extends the measured Arrows Gaussian-tail
U-Net to VP, bridges and all affine-FM interfaces without a family fallback.
"""
import math
import torch
from torch import nn

from models.image_gaussian_fields import ImageUNet
from models.preconditioned_field import PreconditionedField


def to_image(flat,shape,layout):
    h,w,c=shape
    return flat.reshape(-1,h,w,c).permute(0,3,1,2) if layout=='nhwc' else flat.reshape(-1,c,h,w)


def from_image(image,layout):
    return (image.permute(0,2,3,1) if layout=='nhwc' else image).reshape(len(image),-1)


class SharedImagePosteriorField(PreconditionedField):
    def __init__(self,architecture,family,training_config):
        nn.Module.__init__(self)
        self.config=architecture
        self.ambient_dim=architecture.ambient_dim
        self.family=family
        self.training_config=training_config
        if math.prod(training_config.image_shape)!=self.ambient_dim:
            raise ValueError('image shape differs from ambient dimension')
        if training_config.image_training_bf16:
            raise ValueError('the common image protocol uses float32 training for every family')
        self.core=ImageUNet(training_config.image_shape[-1],training_config.image_width)

    def posterior(self,inputs,condition):
        condition=torch.as_tensor(condition,dtype=inputs.dtype,device=inputs.device)
        if condition.ndim==0:condition=condition.expand(len(inputs))
        scale,alpha=self.channel(condition)
        canonical=inputs/alpha[:,None]
        inverse=torch.rsqrt(1+scale.square())[:,None]
        cfg=self.training_config
        image=to_image(canonical*inverse,cfg.image_shape,cfg.image_layout)
        residual=from_image(self.core(image,scale.log()),cfg.image_layout)
        if cfg.field_residual_scaling=='gaussian_tail_v1':
            factor=scale[:,None]*inverse.square()
        elif cfg.field_residual_scaling=='noise_v1':
            factor=scale[:,None]*inverse
        else:
            raise ValueError('unsupported image residual coefficient')
        return canonical*inverse.square()+factor*residual
