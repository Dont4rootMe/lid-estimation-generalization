"""Antithetic Gaussian corruption with an unchanged marginal native objective.

The sampler and RNG consume exactly their original draws. In training only,
each first-half clean point/scale is evaluated at +Z and -Z, for the original
number of network examples. Validation retains its original independent draws.
"""
import torch


def paired_corruption(model,clean,condition,noise):
    if getattr(model,'_lid_noise_pairing','iid') != 'antithetic_v1' or not model.training:
        return clean,condition,noise
    if len(clean)%2:
        raise ValueError('antithetic training requires an even batch')
    half=len(clean)//2
    return (torch.cat((clean[:half],clean[:half]),0),
            torch.cat((condition[:half],condition[:half]),0),
            torch.cat((noise[:half],-noise[:half]),0))
