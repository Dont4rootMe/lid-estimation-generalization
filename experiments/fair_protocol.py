"""Resolve and audit a common comparison independently of observed LID scores."""
import yaml
from dataclasses import asdict,replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import torch

from models import training
from models.image_gaussian_fields import ImageUNet
from models.normalizing_flow import ConditionalFlowConfig,ScaleConditionedRealNVP
from models.spectral_residual_field import SpectralResidualCore

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_PATH=ROOT/'configs/fair_comparison/protocol.yaml'
CONTRACTS_PATH=ROOT/'configs/lambda_repair/native_contracts.yaml'
COMMON_TRAINING_KEYS=('seed','steps','batch_size','learning_rate','weight_decay',
    'gradient_clip_norm','validation_interval_steps','warmup_steps','noise_pairing',
    'terminal_decay_steps','terminal_learning_rate_ratio','ema_decay','ema_start_step',
    'optimizer_schedule_policy','training_target','normalize','normalization_epsilon',
    'image_training_bf16')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def protocol():
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def native_contracts():
    return {c['variant_id']:c['model'] for c in yaml.safe_load(CONTRACTS_PATH.read_text())['model_contracts']}


def geometry(dataset,representation,feature_shape):
    shape=tuple(feature_shape)
    if representation=='coefficients' or dataset.endswith('_pca') or '_pca_' in dataset:
        return dict(kind='vector',ambient_dim=math.prod(shape),feature_shape=list(shape),image_shape=None,image_layout='nhwc')
    if representation!='dataset' or len(shape)!=3:
        raise ValueError('representation has no explicit common architecture rule')
    if shape[-1] in (1,3) and shape[0]>3:
        image_shape=shape;layout='nhwc'
    elif shape[0] in (1,3) and shape[-1]>3:
        image_shape=(shape[1],shape[2],shape[0]);layout='nchw'
    else:
        raise ValueError('ambiguous image channel layout')
    if any(s%4 for s in image_shape[:2]):
        raise ValueError('spatial dimensions must be divisible by four')
    return dict(kind='image',ambient_dim=math.prod(shape),feature_shape=list(shape),image_shape=list(image_shape),image_layout=layout)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


@lru_cache(maxsize=None)
def capacities(kind,dimension):
    """dimension is fitted active rank for vectors, channel count for images."""
    with torch.device('meta'):
        target=parameter_count(SpectralResidualCore(dimension,width=512) if kind=='vector' else ImageUNet(dimension,32))
        candidates=[]
        widths=range(32,2049,32) if kind=='vector' else range(4,65)
        for width in widths:
            if kind=='vector':
                model=ScaleConditionedRealNVP(ConditionalFlowConfig(ambient_dim=dimension,
                    hidden_dim=width,num_coupling_layers=8,conditioner_depth=2,condition_dim=128,
                    fourier_features=32,max_condition_frequency=1.,dropout=0.,log_scale_limit=2.))
                actual=parameter_count(model)
            else:
                actual=8*parameter_count(ImageUNet(dimension,width,output_channels=2*dimension))
            candidates.append((abs(actual-target)/target,width,actual))
    gap,width,actual=min(candidates)
    if gap>protocol()['nf']['maximum_relative_parameter_gap']:
        raise ValueError('no NF capacity match within the declared tolerance')
    return dict(reference_parameters=target,nf_parameters=actual,nf_width=width,nf_relative_gap=(actual-target)/target)


def resolve(variant,geo,*,rank=None,device='cuda',steps=None,preflight=False):
    rules=protocol();base=native_contracts()[variant]
    budget=rules['training']['steps'] if steps is None else steps
    if not preflight and budget!=rules['training']['steps']:
        raise ValueError('benchmark runs use the frozen common budget; short runs must be labelled preflight')
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<2:
        raise ValueError('invalid training budget')
    nf=variant=='scale_conditioned_nf';image=geo['kind']=='image'
    if not image and (rank is None or not 2<=rank<=geo['ambient_dim']):
        raise ValueError('fit the common training covariance before resolving vector/NF capacity')
    cap=capacities(geo['kind'],geo['image_shape'][-1] if image else rank)
    terminal=min(rules['training']['terminal_decay_steps'],max(1,budget//4))
    keys={k:v for k,v in rules['training'].items() if k in training.TrainingConfig.__dataclass_fields__}
    keys.update(device=device,steps=budget,terminal_decay_steps=terminal,ema_start_step=budget-terminal,
        validation_interval_steps=min(500,budget),num_workers=0,training_target='sample_v1',
        field_projection_rank=None if image else rank,
        field_preconditioning='image_gaussian_v1' if image else 'covariance_span_v1',
        field_backbone='image_unet_v1' if image else ('bottleneck_v1' if nf else 'spectral_residual_v1'),
        field_residual_scaling='noise_v1' if nf or not image else 'gaussian_tail_v1',
        field_residual_width=512,image_training_bf16=False,
        image_shape=tuple(geo['image_shape']) if image else None,image_layout=geo['image_layout'],
        image_width=(cap['nf_width'] if nf else 32) if image else 32)
    if nf:
        keys.update(hidden_dim=cap['nf_width'],max_condition_frequency=1.,num_coupling_layers=8,
                    conditioner_depth=2,time_embedding_dim=128)
    config=replace(training.TrainingConfig.from_mapping(base['training']),**keys)
    return config,dict(**cap,actual_parameters=cap['nf_parameters'] if nf else cap['reference_parameters'],
        geometry=geo,rank=rank,kind='preflight' if preflight else 'benchmark',
        protocol_id=rules['id'],protocol_sha256=digest(rules),
        common_training={k:getattr(config,k) for k in COMMON_TRAINING_KEYS})


def build_model(variant,config,ambient_dim):
    family=training._canonical_family(native_contracts()[variant]['family'])
    if variant=='scale_conditioned_nf':
        from models.preconditioned_nf import build_nf
        return build_nf(training._nf_architecture_from_training_config(config,ambient_dim=ambient_dim),config)
    from models.preconditioned_field import build_bottleneck
    architecture=(training._vp_architecture_from_training_config(config,ambient_dim=ambient_dim)
        if family=='vp_diffusion' else training._bottleneck_architecture_from_training_config(config,family=family,ambient_dim=ambient_dim))
    return build_bottleneck(architecture,family,config)


def source_identity():
    paths=[p for folder in ('models','experiments','datasets','utils')
           for p in sorted((ROOT/folder).glob('*.py'))]+sorted((ROOT/'configs').rglob('*.yaml'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def audit_group(rows):
    """Reject mixed budgets, geometry, preprocessing and missing model arms."""
    expected=set(native_contracts())
    if len(rows)!=len(expected) or {r['variant'] for r in rows}!=expected:
        raise ValueError('comparison must contain each native interface exactly once')
    first=rows[0]['resolved']
    for row in rows:
        item=row['resolved']
        for key in ('protocol_sha256','common_training','geometry','rank','kind','reference_parameters'):
            if item[key]!=first[key]:raise ValueError('mixed comparison policy: '+key)
        if row['variant']=='scale_conditioned_nf':
            if abs(item['actual_parameters']/item['reference_parameters']-1)>.10:
                raise ValueError('NF capacity outside tolerance')
        elif item['actual_parameters']!=item['reference_parameters']:
            raise ValueError('vector backbones do not match')
    return True
