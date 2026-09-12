import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from experiments.fair_protocol import (geometry,native_contracts,active_native_contracts,resolve,build_model,
    parameter_count,audit_group,protocol,digest)
from experiments.fair_campaign import matrix,select_scale,validate_reference,prediction_spec,aggregate
from experiments import global_campaign_v2 as v2
from experiments.lambda_repair_eval import CanonicalPosterior
from models import training
from models.image_gaussian_fields import NativeGaussianImageField,ImageFieldConfig

torch.set_num_threads(2)


def image_geo(channels=1):
    return geometry('image_fixture','dataset',[4,4,channels])


def test_complete_matrix_has_no_family_specific_architecture_fallback():
    result=matrix()
    assert result['cells']==39 and result['trainings']==429
    for key in {r['cell_key'] for r in result['rows']}:
        rows=[r for r in result['rows'] if r['cell_key']==key]
        assert {r['variant'] for r in rows}==set(active_native_contracts())
        assert len({r['architecture'] for r in rows if r['variant']!='scale_conditioned_nf'})==1
    arrows=[r for r in result['rows'] if r['cell_key']=='e2/e2_arrows/dataset']
    assert all('U-Net' in r['architecture'] for r in arrows)


@pytest.mark.parametrize('dimension',[2,3,6,20,30])
def test_full_ambient_nf_matches_actual_shared_vector_capacity(dimension):
    geo=geometry('fixture','coefficients',[dimension]);rows=[]
    for variant in active_native_contracts():
        cfg,receipt=resolve(variant,geo,device='cpu',steps=8,preflight=True)
        with torch.device('meta'):model=build_model(variant,cfg,dimension)
        assert cfg.field_projection_rank is None and not hasattr(model,'basis')
        assert parameter_count(model)==receipt['actual_parameters']
        rows.append(dict(variant=variant,resolved=receipt))
    assert audit_group(rows)
    assert abs(rows[0]['resolved']['nf_relative_gap'])<=.1
    bad=copy.deepcopy(rows);bad[0]['resolved']['common_training']['ema_decay']=None
    with pytest.raises(ValueError,match='mixed comparison'):audit_group(bad)
    with pytest.raises(ValueError,match='exactly once'):audit_group(rows[:-1])


@pytest.mark.parametrize('channels',[1,3])
def test_image_capacity_and_training_policy_are_common(channels):
    rows=[]
    for variant in active_native_contracts():
        cfg,receipt=resolve(variant,image_geo(channels),device='cpu',steps=8,preflight=True)
        with torch.device('meta'):model=build_model(variant,cfg,16*channels)
        assert parameter_count(model)==receipt['actual_parameters']
        assert cfg.optimizer_schedule_policy=='shared_terminal_v1'
        assert cfg.noise_pairing=='iid' and not cfg.image_training_bf16
        rows.append(dict(variant=variant,resolved=receipt))
    assert audit_group(rows)
    assert rows[0]['resolved']['nf_width']==5
    assert rows[0]['resolved']['nf_coupling_layers']==2
    assert rows[0]['resolved']['nf_capacity_match_required']
    assert abs(rows[0]['resolved']['nf_relative_gap'])<=.1


def test_image_native_heads_expose_raw_core_without_posterior_wrapper():
    torch.manual_seed(9);x=torch.randn(2,16,dtype=torch.float64)
    reference=None
    for variant,contract in native_contracts().items():
        if variant in ('scale_conditioned_nf','pfgmpp'):continue
        cfg,_=resolve(variant,image_geo(),device='cpu',steps=8,preflight=True)
        model=build_model(variant,cfg,16).double().eval()
        if reference is None:
            with torch.no_grad():model.core.output.weight.normal_(std=.01)
            reference=copy.deepcopy(model.core.state_dict())
        else:model.core.load_state_dict(reference)
        for condition in [.1,.2,.3]:
            z=x.clone().requires_grad_(True)
            expected=model.core(z.reshape(-1,4,4,1).permute(0,3,1,2),
                z.new_full((len(z),),condition)).permute(0,2,3,1).reshape_as(z)
            if variant=='schrodinger_bridge':expected=expected/cfg.dsb_step_size
            actual=model(z,condition)
            torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-10)
            direction=torch.arange(16,dtype=z.dtype).expand_as(z)/16
            g1=torch.autograd.grad(actual,z,direction,retain_graph=True)[0]
            g2=torch.autograd.grad(expected,z,direction)[0]
            torch.testing.assert_close(g1,g2,rtol=3e-10,atol=3e-10)


@pytest.mark.parametrize('variant,family',[('ve_diffusion','gaussian_diffusion'),('rectified_flow','rectified_flow')])
def test_native_image_objective_has_no_common_posterior_call(variant,family):
    cfg,_=resolve(variant,image_geo(),device='cpu',steps=8,preflight=True)
    new=build_model(variant,cfg,16).double()
    with torch.no_grad():new.core.output.weight.normal_(std=.01)
    x=torch.randn(8,16,dtype=torch.float64)
    def forbidden(*args,**kwargs):raise AssertionError('training accessed the evaluation-only posterior adapter')
    new.canonical_posterior=forbidden
    a=training._objective(family,new,x,cfg,torch.Generator().manual_seed(31))
    ga=torch.autograd.grad(a,tuple(new.parameters()))
    assert torch.isfinite(a) and all(torch.isfinite(g).all() for g in ga)


def test_spatial_nf_nontrivial_inverse_and_full_jacobian_determinant():
    cfg,_=resolve('scale_conditioned_nf',image_geo(),device='cpu',steps=8,preflight=True)
    model=build_model('scale_conditioned_nf',cfg,16).double()
    with torch.no_grad():
        for c in model.couplings:
            c.conditioner.output.weight.normal_(std=.002)
            c.conditioner.output.bias.normal_(std=.002)
    z=torch.randn(2,16,dtype=torch.float64);scale=torch.tensor([.2,2.3],dtype=z.dtype)
    x,forward=model.decode(z,scale);back,inverse=model.encode(x,scale)
    torch.testing.assert_close(z,back,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(forward,-inverse,rtol=1e-10,atol=1e-10)
    jac=torch.autograd.functional.jacobian(lambda y:model.encode(y[None],.2)[0][0],x[0])
    _,exact=torch.linalg.slogdet(jac)
    torch.testing.assert_close(exact,inverse[0],rtol=1e-9,atol=1e-9)
    log_scale=torch.tensor(.2,dtype=x.dtype).log().requires_grad_(True)
    value=model.log_prob(x,log_scale.exp()).sum()
    derivative=torch.autograd.grad(value,log_scale)[0]
    h=1e-5
    fd=(model.log_prob(x,(log_scale+h).exp()).sum()-model.log_prob(x,(log_scale-h).exp()).sum())/(2*h)
    torch.testing.assert_close(derivative,fd,rtol=1e-7,atol=1e-7)


def test_reference_cannot_be_taken_from_another_method_or_protocol():
    cell=SimpleNamespace(suite_id='e1',reference_dataset='e1_sampled_fmnist_step1',representation='dataset')
    manifest=dict(variant='ve_diffusion',protocol_sha256='a',source_sha256={'x':'b'},kind='benchmark')
    ref=dict(manifest,cell_key='e1/e1_sampled_fmnist_step1/dataset',status='complete',
        selected_lambda=float(v2.unknown_reference_lambdas()[20]),selection=dict(status='selected'))
    validate_reference(ref,manifest,cell)
    for key,value in [('variant','vp_diffusion'),('protocol_sha256','old'),('kind','preflight')]:
        with pytest.raises(ValueError):validate_reference(dict(ref,**{key:value}),manifest,cell)


def test_benchmark_budget_cannot_be_shortened_without_preflight_label():
    with pytest.raises(ValueError,match='frozen common budget'):
        resolve('ve_diffusion',image_geo(),steps=8)
    image=geometry('e6_exp_pca','dataset',[1,28,28])
    assert image['kind']=='image'
    config,_=resolve('scale_conditioned_nf',image)
    assert config.field_projection_rank is None and config.field_backbone=='image_unet_v1'
    with pytest.raises(ValueError,match='forbids reduced-rank'):
        resolve('scale_conditioned_nf',geometry('e6_exp_pca','coefficients',[30]),rank=3)


def test_nf_readout_uses_likelihood_derivative_for_both_geometries():
    for geo in (image_geo(),geometry('e6_exp_pca','coefficients',[30])):
        spec=prediction_spec('scale_conditioned_nf',geo)
        assert spec['derivative_backend']=='exact' and spec['trace_probes']==0


def test_unknown_dependent_cell_cannot_retune_reference_scale():
    cell=SimpleNamespace(target_policy='sample_size',dataset='step2',reference_dataset='step1')
    grid=np.geomspace(1/256,64,50)
    selected=float(v2.unknown_reference_lambdas()[20])
    reference=dict(selected_lambda=selected,selection=dict(status='selected'))
    for curve in [np.ones((8,50)),np.random.default_rng(9).normal(size=(8,50))]:
        scale,receipt=select_scale(curve,grid,None,cell,reference)
        assert scale==selected and receipt['criterion']=='reuse_reference_mean_kneedle'


def test_shared_vp_really_trains_with_common_clipping_decay_and_ema(tmp_path):
    cfg,_=resolve('vp_diffusion',image_geo(),device='cpu',steps=4,preflight=True)
    x=np.random.default_rng(23).normal(size=(40,16)).astype(np.float32)
    result=training.train_model('vp_diffusion',x[:32],x[32:],cfg,tmp_path/'vp.pt')
    loaded=training.load_checkpoint(tmp_path/'vp.pt',device='cpu')
    assert result.metrics['steps_completed']==4
    assert result.history[-1].learning_rate==pytest.approx(.0002*.01)
    assert result.config.gradient_clip_norm==1. and result.config.ema_decay==.999
    for key,value in result.model.state_dict().items():
        torch.testing.assert_close(value,loaded.model.state_dict()[key],rtol=0,atol=0)


@pytest.mark.parametrize('case', ['empty', 'historical', 'preflight'])
def test_aggregate_rejects_unfinished_or_incompatible_evidence(tmp_path, case):
    if case == 'empty':
        expected = 'incomplete fair matrix'
    elif case == 'historical':
        (tmp_path/'complete.json').write_text(json.dumps({'status': 'complete'}))
        expected = 'unrecognized historical result'
    else:
        route = matrix()['rows'][0]
        (tmp_path/'complete.json').write_text(json.dumps(dict(
            protocol_sha256=digest(protocol()), cell_key=route['cell_key'],
            variant=route['variant'], kind='preflight', steps_completed=20)))
        expected = 'preflight or incomplete budget'
    with pytest.raises(ValueError, match=expected):
        aggregate(tmp_path)
