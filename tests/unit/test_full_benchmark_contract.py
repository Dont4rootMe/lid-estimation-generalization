"""Density identities, shared selection, and adversarial data-order controls."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
import torch

from experiments import fair_data, fair_measurements as measurement, fair_outputs
from experiments.fair_protocol import native_contracts,resolve,geometry,build_model
from experiments.fair_campaign import select_scale
from models import training
from models.neural_fields import exact_divergence
from models.student_density import density_dilation


@pytest.mark.parametrize('ambient',[2,6,30])
@pytest.mark.parametrize('df',[5,128])
@pytest.mark.parametrize('scale',[.02,.7,3.])
def test_student_full_equals_direct_mixture_density_derivative(ambient,df,scale):
    generator=torch.Generator().manual_seed(84011)
    centers=torch.randn(7,ambient,generator=generator,dtype=torch.float64)
    prior=torch.arange(1,8,dtype=torch.float64)
    query=torch.randn(3,ambient,generator=generator,dtype=torch.float64)
    def posterior(q,lam):
        lam=torch.as_tensor(lam,dtype=q.dtype).expand(len(q))
        distance=(q[:,None]-centers).square().sum(2)
        return (prior.log()-(ambient+df)/2*torch.log1p(
            distance/((df-2)*lam[:,None].square()))).softmax(1)@centers
    response=exact_divergence(posterior,query,scale)
    full,_,_=density_dilation(posterior,query,scale,response,df)
    # Differentiate the scalar mixture density directly, without the identity.
    log_scale=torch.full((len(query),),np.log(scale),dtype=query.dtype,requires_grad=True)
    distance=(query[:,None]-centers).square().sum(2)
    log_density=-ambient*log_scale+torch.logsumexp(prior.log()-(ambient+df)/2*
        torch.log1p(distance/((df-2)*log_scale[:,None].exp().square())),1)
    direct=ambient+torch.autograd.grad(log_density.sum(),log_scale)[0]
    torch.testing.assert_close(full,direct,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('ambient',[2,30,784])
@pytest.mark.parametrize('df',[5,128])
def test_student_full_recovers_half_line_dimension_at_boundary(ambient,df):
    p=(ambient+df)/2
    integrate=lambda f:quad(f,0,np.inf,epsabs=1e-12)[0]
    mass=integrate(lambda u:(1+u*u)**(-p))
    mean=integrate(lambda u:u*(1+u*u)**(-p))/mass
    score=integrate(lambda u:2*p*u*(1+u*u)**(-p-1))/mass
    response=integrate(lambda u:2*p*u*u*(1+u*u)**(-p-1))/mass-mean*score
    # Exact half-line posterior at y=0 has b=lambda*sqrt(nu-2)*mean.
    def posterior(q,scale):
        unit=torch.zeros_like(q);unit[:,0]=1
        return unit*scale[:,None]*np.sqrt(df-2)*mean
    full,_,_=density_dilation(posterior,torch.zeros(1,ambient,dtype=torch.float64),.7,
        torch.tensor([response],dtype=torch.float64),df)
    assert full.item()==pytest.approx(1,abs=1e-8)
    assert response<.6  # A response-only replacement would fail this control.


@pytest.mark.parametrize('variant',list(native_contracts()))
def test_all_native_interfaces_select_the_full_density_quantity(variant):
    primary=fair_outputs.readouts(variant)[0]
    assert primary==('ols5' if variant=='scale_conditioned_nf' else 'full')
    if variant!='scale_conditioned_nf':
        assert native_contracts()[variant]['primary_readout']=='full'
    grid=measurement.common_scales();target=np.ones(3)
    full=target[:,None]+np.log2(grid)[None,:]**2
    scale,receipt=select_scale(full,grid,target,SimpleNamespace(target_policy='known_lid'))
    assert scale==1 and receipt['selection_quantity']=='full_density_lid'


def test_response_cannot_retune_supervised_or_pointwise_full_scale():
    grid=measurement.common_scales();legacy=measurement.legacy_scales();target=np.ones(3)
    holdout=target[:,None]+np.log2(grid)[None,:]**2
    plan=measurement.known_plan(holdout,grid,target)
    # Full has a knee in two rows and no knee in a constant third row.
    common=np.tile(30/(1+grid**2),(3,1));common[-1]=4
    old=np.tile(30/(1+legacy**2),(3,1));old[-1]=4
    response=np.tile(1+np.log2(grid/32)**2,(3,1))
    response_old=np.tile(1+np.log2(legacy/32)**2,(3,1))
    a,arrays=measurement.known_results(common,old,target,plan,30,
        response_curves=(response,response_old))
    b,other=measurement.known_results(common,old,target,plan,30,
        response_curves=(response[:,::-1],response_old[:,::-1]))
    assert a['supervised_common_grid']['selected_lambda']==b['supervised_common_grid']['selected_lambda']==1
    assert np.argmin(abs(response[0]-1))!=plan['supervised_common_grid']['selected_index']
    for name,values in [('kneedle_common_grid',response),('kneedle_legacy',response_old)]:
        np.testing.assert_array_equal(arrays[name+'_index'],other[name+'_index'])
        assert a[name]['selected_lambda_counts']==b[name]['selected_lambda_counts']
        indices=arrays[name+'_index'];valid=np.flatnonzero(indices>=0)
        assert len(valid)>0
        np.testing.assert_array_equal(arrays[name+'_response_prediction'][valid],values[valid,indices[valid]])
        assert arrays[name+'_response_prediction'][-1]==30


def test_within_class_permutation_fails_pinned_input_check(tmp_path,monkeypatch):
    features=np.arange(12,dtype=np.float32).reshape(4,3)
    labels=np.array([1,1,2,2])
    np.save(tmp_path/'dataset.npy',features);np.save(tmp_path/'labels.npy',labels)
    paths={name:tmp_path/(name+'.npy') for name in ('dataset','labels')}
    cell=dict(suite_id='e5',dataset='fixture',representation='dataset',reference_dataset='fixture')
    key='e5/fixture/dataset'
    pinned=dict(cells={key:cell},datasets={'fixture':{'files':{
        'fixture/test/'+p.name:dict(sha256=fair_data.file_sha(p),size_bytes=p.stat().st_size)
        for p in paths.values()}}})
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text(json.dumps(pinned))
    monkeypatch.setattr(fair_data,'MANIFEST_PATH',manifest_path)
    loaded=SimpleNamespace(source_paths=paths)
    hashes=fair_data.validate_loaded(cell,loaded,'test')
    row=dict(cell=cell,input_manifest_sha256=fair_data.contract_sha(),
        query_order_contract='pinned_input_rows_v1',test_files_sha256=hashes)
    fair_data.validate_receipt(row,splits=('test',))
    # Same labels, row numbers, shapes, sizes and value distribution: wrong pairs.
    np.save(paths['dataset'],features[[1,0,3,2]])
    np.testing.assert_array_equal(np.load(paths['labels']),labels)
    with pytest.raises(ValueError,match='input files/order'):
        fair_data.validate_loaded(cell,loaded,'test')
    forged=dict(row,test_files_sha256={p.name:fair_data.file_sha(p) for p in paths.values()})
    with pytest.raises(ValueError,match='input files/order'):
        fair_data.validate_receipt(forged,splits=('test',))
    missing=dict(row);missing.pop('input_manifest_sha256')
    with pytest.raises(ValueError,match='attestation'):
        fair_data.validate_receipt(missing,splits=('test',))


@pytest.mark.parametrize('variant',['t_flowmatching','pfgmpp'])
def test_legacy_student_weights_migration_preserves_all_training_fields(variant):
    config,_=resolve(variant,geometry('fixture','coefficients',[3]),device='cpu',steps=4,preflight=True)
    expected=training._model_contract(native_contracts()[variant]['family'],config)
    old=dict(expected,schema_version=1,primary_readout='response');old.pop('full_readout')
    assert training._compatible_model_contract(old,expected)
    altered=dict(old,kernel_df=old['kernel_df']+1)
    assert not training._compatible_model_contract(altered,expected)


def test_canonical_eight_interface_losses_and_gradients_are_equivalent():
    variants=['ve_diffusion','schrodinger_bridge','direct_rectified_flow','posterior_rectified_flow',
        'direct_log_noise_affine_flow','posterior_log_noise_affine_flow',
        'direct_vp_trigonometric_flow','posterior_vp_trigonometric_flow']
    x=torch.randn(6,3,dtype=torch.float64,generator=torch.Generator().manual_seed(195))
    state=None;reference_loss=None;reference_gradient=None
    for variant in variants:
        torch.manual_seed(191)
        config,_=resolve(variant,geometry('fixture','coefficients',[3]),device='cpu',steps=4,preflight=True)
        model=build_model(variant,config,3).double().train();model._lid_noise_pairing='antithetic_v1'
        if state is None:
            with torch.no_grad():
                model.core.output.weight.normal_(0,.01);model.core.output.bias.normal_(0,.01)
            state=copy.deepcopy(model.core.state_dict())
        else:model.core.load_state_dict(state)
        family=training._canonical_family(native_contracts()[variant]['family'])
        loss=training._objective(family,model,x,config,torch.Generator().manual_seed(199))
        gradient=torch.cat([g.flatten() for g in torch.autograd.grad(loss,tuple(model.parameters()))])
        if reference_loss is None:reference_loss=loss.detach();reference_gradient=gradient
        torch.testing.assert_close(loss,reference_loss,atol=1e-12,rtol=1e-12)
        assert (gradient-reference_gradient).norm()/reference_gradient.norm()<1e-10
