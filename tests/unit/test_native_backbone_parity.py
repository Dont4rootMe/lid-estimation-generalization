"""Prevent separate field backbones and enforce the revised NF capacity match."""
import copy

import pytest
import torch

from experiments import fair_protocol as fair
from models.image_gaussian_fields import ImageUNet
from models.point_residual_field import PointResidualCore


@pytest.mark.parametrize('shape,dimension,field_count,nf_count',[
    ([1,28,28],784,97905,91300),
    ([32,32,3],3072,98195,91848),
])
def test_every_image_field_uses_the_same_stock8_core_and_matched_nf(shape,dimension,field_count,nf_count):
    geo=fair.geometry('full_image_fixture','dataset',shape)
    reference=None;rows=[]
    for variant in fair.native_contracts():
        cfg,receipt=fair.resolve(variant,geo,device='cpu',steps=4,preflight=True)
        with torch.device('meta'):
            model=fair.build_model(variant,cfg,dimension)
        assert cfg.field_projection_rank is None
        assert fair.parameter_count(model)==receipt['actual_parameters']
        if variant=='scale_conditioned_nf':
            assert cfg.image_width==5 and len(model.couplings)==2
            assert fair.parameter_count(model)==nf_count
            assert receipt['nf_capacity_match_required']
            assert abs(receipt['nf_relative_gap'])<=.1
            assert receipt['nf_capacity_policy']=='matched_to_shared_image_field'
        else:
            assert type(model.core) is ImageUNet and cfg.image_width==8
            assert model.core.input.out_channels==8
            assert fair.parameter_count(model)==field_count*(2 if variant=='schrodinger_bridge' else 1)
            signature={k:tuple(v.shape) for k,v in model.core.state_dict().items()}
            if reference is None:reference=signature
            assert signature==reference
        rows.append(dict(variant=variant,resolved=receipt))
    assert fair.audit_group(rows,include_disabled=True)
    bad=copy.deepcopy(rows)
    next(r for r in bad if r['variant']=='t_flowmatching')['resolved']['common_field_backbone']['width']=32
    with pytest.raises(ValueError,match='mixed comparison'):
        fair.audit_group(bad,include_disabled=True)
    bad=copy.deepcopy(rows)
    next(r for r in bad if r['variant']=='scale_conditioned_nf')['resolved']['actual_parameters']=field_count
    with pytest.raises(ValueError,match='NF parameters'):
        fair.audit_group(bad,include_disabled=True)


def test_full_vector_core_is_shared_with_both_new_families():
    geo=fair.geometry('full_vector_fixture','coefficients',[30])
    reference=None
    for variant in fair.native_contracts():
        if variant=='scale_conditioned_nf':continue
        cfg,receipt=fair.resolve(variant,geo,device='cpu',steps=4,preflight=True)
        with torch.device('meta'):model=fair.build_model(variant,cfg,30)
        assert type(model.core) is PointResidualCore
        assert model.core.input.in_features==30
        assert not any('spatial_frequencies' in key for key in model.state_dict())
        assert cfg.field_residual_width==512 and cfg.field_projection_rank is None
        assert len(model.core.blocks)==4 and model.core.output.out_features==30
        assert fair.parameter_count(model)==receipt['actual_parameters']==2698014*(2 if variant=='schrodinger_bridge' else 1)
        signature={k:tuple(v.shape) for k,v in model.core.state_dict().items()}
        if reference is None:reference=signature
        assert signature==reference


@pytest.mark.parametrize('kind,new_width',[('image',12),('vector',384)])
def test_declared_width_changes_the_model_and_capacity_cache_together(monkeypatch,kind,new_width):
    rules=copy.deepcopy(fair.protocol())
    geo=fair.geometry('fixture','dataset',[1,28,28]) if kind=='image' else fair.geometry('fixture','coefficients',[30])
    dimension=784 if kind=='image' else 30
    _,before=fair.resolve('ve_diffusion',geo,device='cpu',steps=4,preflight=True)
    rules[kind+'_data']['width']=new_width
    monkeypatch.setattr(fair,'protocol',lambda:rules)
    for variant in ['ve_diffusion','rectified_flow','t_flowmatching','pfgmpp']:
        cfg,after=fair.resolve(variant,geo,device='cpu',steps=4,preflight=True)
        with torch.device('meta'):model=fair.build_model(variant,cfg,dimension)
        assert fair.parameter_count(model)==after['actual_parameters']!=before['actual_parameters']
        assert (cfg.image_width if kind=='image' else cfg.field_residual_width)==new_width
        if kind=='image':
            assert after['nf_parameters']!=before['nf_parameters']
            assert abs(after['nf_relative_gap'])<=.1


def test_nf_policy_rejects_candidates_without_a_capacity_match(monkeypatch):
    rules=copy.deepcopy(fair.protocol());rules['nf']['image_width_candidates']=[32]
    monkeypatch.setattr(fair,'protocol',lambda:rules)
    with pytest.raises(ValueError,match='no NF capacity match'):
        fair.resolve('t_flowmatching',fair.geometry('fixture','dataset',[1,28,28]))
