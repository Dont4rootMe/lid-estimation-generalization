"""Exactly invertible image RealNVP with spatial U-Net conditioners.

The conditioners share the vector field's architecture template, with two
outputs per channel for affine scale/translation. Their width is selected by
parameter count alone. The complete density remains a flow, not a U-Net density
proxy. All determinants include the common Gaussian input preconditioner.
"""
import math
import torch
from torch import nn

from models.image_gaussian_fields import ImageUNet
from models.normalizing_flow import ScaleConditionedRealNVP
from models.shared_image_field import to_image,from_image


class ImageCoupling(nn.Module):
    def __init__(self,shape,width,parity,limit,*,scalar_condition=False):
        super().__init__()
        h,w,c=shape
        mask=((torch.arange(h)[:,None]+torch.arange(w)[None,:]+parity)%2).float()
        self.register_buffer('mask',mask[None,None])
        self.conditioner=ImageUNet(c,width,output_channels=2*c,scalar_condition=scalar_condition)
        self.limit=limit

    def parameters_at(self,image,log_scale):
        raw,shift=self.conditioner(image*self.mask,log_scale).chunk(2,dim=1)
        active=1-self.mask
        return self.limit*torch.tanh(raw)*active,shift*active

    def forward(self,image,log_scale,inverse=False):
        log_gain,shift=self.parameters_at(image,log_scale)
        if inverse:
            return (image-shift)*torch.exp(-log_gain),-log_gain.flatten(1).sum(1)
        return image*torch.exp(log_gain)+shift,log_gain.flatten(1).sum(1)


class ImageConditionedRealNVP(ScaleConditionedRealNVP):
    def __init__(self,architecture,training_config):
        nn.Module.__init__(self)
        self.config=architecture
        self.training_config=training_config
        if math.prod(training_config.image_shape)!=architecture.ambient_dim:
            raise ValueError('NF image shape differs from ambient dimension')
        if training_config.image_training_bf16:
            raise ValueError('common image NF uses float32 training')
        self.couplings=nn.ModuleList([
            ImageCoupling(training_config.image_shape,training_config.image_width,j%2,architecture.log_scale_limit,
                          scalar_condition=architecture.fourier_features == 0)
            for j in range(architecture.num_coupling_layers)])

    def scale(self,epsilon,reference):
        value=torch.as_tensor(epsilon,device=reference.device,dtype=reference.dtype)
        if value.ndim==0:value=value.expand(len(reference))
        elif value.ndim==2 and value.shape[1]==1:value=value[:,0]
        if value.shape==(1,):value=value.expand(len(reference))
        if value.shape!=(len(reference),) or not torch.isfinite(value).all() or torch.any(value<=0):
            raise ValueError('epsilon must be positive and match the query batch')
        return value

    def encode(self,observations,epsilon):
        flat=self._flatten(observations)
        scale=self.scale(epsilon,flat)
        standard=torch.sqrt(1+scale.square())
        cfg=self.training_config
        image=to_image(flat/standard[:,None],cfg.image_shape,cfg.image_layout)
        determinant=-self.config.ambient_dim*standard.log()
        for coupling in reversed(self.couplings):
            image,contribution=coupling(image,scale.log(),inverse=True)
            determinant=determinant+contribution
        return from_image(image,cfg.image_layout),determinant

    def decode(self,latent,epsilon):
        flat=self._flatten(latent)
        scale=self.scale(epsilon,flat)
        cfg=self.training_config
        image=to_image(flat,cfg.image_shape,cfg.image_layout)
        standard=torch.sqrt(1+scale.square())
        determinant=self.config.ambient_dim*standard.log()
        for coupling in self.couplings:
            image,contribution=coupling(image,scale.log())
            determinant=determinant+contribution
        return from_image(image,cfg.image_layout)*standard[:,None],determinant
