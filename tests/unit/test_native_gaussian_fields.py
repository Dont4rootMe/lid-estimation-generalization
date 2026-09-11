from dataclasses import replace

import numpy as np
import pytest
import torch

from models.gaussian_fields import NativeGaussianBottleneck
from models.training import TrainingConfig, load_checkpoint, predict_lid, train_model


@pytest.mark.parametrize('family', ['gaussian_diffusion', 'rectified_flow'])
def test_loss_all_parameter_gradients_and_analytic_jacobian(family):
    torch.manual_seed(371)
    model = NativeGaussianBottleneck(12, (8,4), 8, native_family=family,
        condition_transform='log' if family == 'gaussian_diffusion' else 'linear').double()
    x, z = torch.randn(9,12,dtype=torch.float64), torch.randn(9,12,dtype=torch.float64)
    lam = torch.logspace(-8,6,9,base=2,dtype=torch.float64)
    native = lam if family == 'gaussian_diffusion' else 1/(1+lam)
    a,b,inv = model.coefficients(x,native)
    for i in (0,4,8):
        jac = torch.autograd.functional.jacobian(
            lambda y:model(y[None],native[i])[0], x[i])
        coefficient = a[i]*inv[i]**2 if family == 'gaussian_diffusion' else (a[i]-b[i])*inv[i]**2
        torch.testing.assert_close(jac, torch.eye(12,dtype=torch.float64)*coefficient)
    torch.nn.init.normal_(model.decoder[-1].weight, std=.04)
    prediction = model(a*x+b*z,native)
    direct = ((prediction-x)/b).square().mean() if family == 'gaussian_diffusion' else (prediction-(x-z)).square().mean()
    stable = model.residual_loss(x,native,z)
    torch.testing.assert_close(stable,direct,rtol=1e-12,atol=1e-12)
    left = torch.autograd.grad(direct,tuple(model.parameters()))
    right = torch.autograd.grad(stable,tuple(model.parameters()))
    for l,r in zip(left,right):
        torch.testing.assert_close(l,r,rtol=1e-10,atol=1e-11)
    assert torch.count_nonzero(right[-2][:,:12]) == 0


@pytest.mark.parametrize('family', ['gaussian_diffusion', 'rectified_flow'])
def test_train_checkpoint_adapter_roundtrip(tmp_path,family):
    config = TrainingConfig(seed=0,device='cpu',training_mode='fixed_steps_v1',steps=4,
        warmup_steps=0,validation_interval_steps=2,batch_size=8,learning_rate=2e-4,
        weight_decay=1e-6,gradient_clip_norm=1,early_stopping_patience=None,
        field_hidden_sizes=(8,4),time_embedding_dim=8,
        native_preconditioning='unit_rms_gaussian_no_input_skip_v1')
    config = replace(config,**({'sigma_min':1/256,'sigma_max':64} if family == 'gaussian_diffusion'
                               else {'time_min':1/65,'time_max':256/257}))
    rng = np.random.default_rng(241)
    fit,hold = rng.normal(size=(32,12)),rng.normal(size=(8,12))
    result = train_model(family,fit,hold,config,tmp_path/'model.pt')
    loaded = load_checkpoint(tmp_path/'model.pt')
    assert isinstance(loaded.model,NativeGaussianBottleneck)
    for name in (('full','response') if family == 'rectified_flow' else ('full',)):
        l = predict_lid(result,hold[:2],.4 if family == 'rectified_flow' else 2,
                        readout=name,divergence_backend='exact')
        r = predict_lid(loaded,hold[:2],.4 if family == 'rectified_flow' else 2,
                        readout=name,divergence_backend='exact')
        np.testing.assert_array_equal(l,r)


def test_wrong_family_rejected():
    from models.training import _model_contract
    config = TrainingConfig(field_hidden_sizes=(8,4),native_preconditioning='unit_rms_gaussian_no_input_skip_v1')
    with pytest.raises(ValueError,match='supports VE'):
        _model_contract('vp_diffusion',config)


@pytest.mark.parametrize('family',['gaussian_diffusion','rectified_flow'])
def test_antithetic_preserves_native_loss_gradient_and_validation(family):
    from models import training
    config=TrainingConfig(batch_size=8,field_hidden_sizes=(8,4),time_embedding_dim=8,
        native_preconditioning='unit_rms_gaussian_no_input_skip_v1',
        native_noise_pairing='antithetic_v1',sigma_min=1/256,sigma_max=64,
        time_min=1/65,time_max=256/257)
    torch.manual_seed(934)
    model=NativeGaussianBottleneck(12,(8,4),8,native_family=family,
        condition_transform='log' if family=='gaussian_diffusion' else 'linear').double()
    torch.nn.init.normal_(model.decoder[-1].weight,std=.02)
    batch=torch.randn(8,12,dtype=torch.float64)
    gen=torch.Generator().manual_seed(373)
    actual=training._objective(family,model,batch,config,gen)
    replay=torch.Generator().manual_seed(373)
    x=batch[:4]
    if family=='gaussian_diffusion':
        condition=training._sample_log_uniform(4,minimum=config.sigma_min,
            maximum=config.sigma_max,data=x,generator=replay)
    else:
        condition=config.time_min+(config.time_max-config.time_min)*torch.rand(4,dtype=x.dtype,generator=replay)
    z=torch.randn(x.shape,dtype=x.dtype,generator=replay)
    a,b,_=model.coefficients(x,condition)
    losses=[]
    for noise in (z,-z):
        pred=model(a*x+b*noise,condition)
        error=(pred-x)/b if family=='gaussian_diffusion' else pred-x+noise
        losses.append(error.square().mean())
    expected=(losses[0]+losses[1])/2
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    for left,right in zip(torch.autograd.grad(actual,tuple(model.parameters())),
                          torch.autograd.grad(expected,tuple(model.parameters()))):
        torch.testing.assert_close(left,right,atol=1e-11,rtol=1e-10)
    independent=replace(config,native_noise_pairing='independent_v1')
    assert training._validation_loss(family,model,batch,config,seed=812)==training._validation_loss(
        family,model,batch,independent,seed=812)


@pytest.mark.parametrize('family',['gaussian_diffusion','rectified_flow'])
def test_image_progress_resume_is_exact_with_default_architecture_fields(tmp_path,monkeypatch,family):
    from models import training
    torch.set_num_threads(2)
    config=TrainingConfig(seed=71,device='cpu',training_mode='fixed_steps_v1',steps=6,
        warmup_steps=0,validation_interval_steps=2,batch_size=4,learning_rate=2e-4,
        early_stopping_patience=None,native_preconditioning='unit_rms_gaussian_image_v1',
        native_noise_pairing='antithetic_v1',image_shape=(4,4,3),image_width=4)
    config=replace(config,**({'sigma_min':1/256,'sigma_max':64} if family=='gaussian_diffusion'
                            else {'time_min':1/65,'time_max':256/257}))
    rng=np.random.default_rng(431)
    fit,hold=rng.normal(size=(16,4,4,3)),rng.normal(size=(4,4,4,3))
    baseline=train_model(family,fit,hold,config,tmp_path/'baseline.pt')
    progress=tmp_path/'progress.pt'
    original=training._atomic_torch_save
    def save_and_interrupt(path,payload):
        if path==progress:
            payload['architecture'].pop('prior')
            payload['architecture'].pop('tail_order')
            original(path,payload)
            raise RuntimeError('simulated interruption after a validated step')
        return original(path,payload)
    monkeypatch.setattr(training,'_atomic_torch_save',save_and_interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        train_model(family,fit,hold,config,tmp_path/'resumed.pt',progress_checkpoint_path=progress)
    monkeypatch.setattr(training,'_atomic_torch_save',original)
    resumed=train_model(family,fit,hold,config,tmp_path/'resumed.pt',progress_checkpoint_path=progress)
    assert resumed.history==baseline.history
    assert resumed.best_epoch==baseline.best_epoch
    for name,value in baseline.model.state_dict().items():
        torch.testing.assert_close(value,resumed.model.state_dict()[name],rtol=0,atol=0)


@pytest.mark.parametrize('family',['gaussian_diffusion','rectified_flow'])
@pytest.mark.parametrize('prior,tail_order',[('unit_gaussian',1),('unit_gaussian',2),('pixel_spike_gaussian_v1',1)])
def test_spatial_field_loss_gradient_and_serialization(tmp_path,family,prior,tail_order):
    from models.image_gaussian_fields import ImageFieldConfig,NativeGaussianImageField
    torch.set_num_threads(2)
    torch.manual_seed(127)
    model = NativeGaussianImageField(ImageFieldConfig(48,(4,4,3),4,False,prior,tail_order),native_family=family).double()
    x,z = torch.randn(3,48,dtype=torch.float64),torch.randn(3,48,dtype=torch.float64)
    lam = torch.tensor([1/256,1,64],dtype=torch.float64)
    native = lam if family == 'gaussian_diffusion' else 1/(1+lam)
    if prior=='pixel_spike_gaussian_v1':
        model.pixel_prior.fit(x,torch.zeros(48,dtype=torch.float64))
    a,b,inv = model.coefficients(x,native)
    initial = model(x,native)
    if prior=='unit_gaussian':
        expected = a*inv.square()*x if family == 'gaussian_diffusion' else (a-b)*inv.square()*x
    else:
        expected = model.pixel_prior.posterior(x/a,b/a)[0]
        if family=='rectified_flow':
            expected=(expected-x)/b
    torch.testing.assert_close(initial,expected)
    torch.nn.init.normal_(model.field.output.weight,std=.02)
    predicted = model(a*x+b*z,native)
    direct = ((predicted-x)/b).square().mean() if family == 'gaussian_diffusion' else (predicted-x+z).square().mean()
    stable = model.residual_loss(x,native,z)
    torch.testing.assert_close(stable,direct,atol=1e-12,rtol=1e-12)
    for l,r in zip(torch.autograd.grad(stable,tuple(model.parameters())),
                   torch.autograd.grad(direct,tuple(model.parameters()))):
        torch.testing.assert_close(l,r,atol=1e-11,rtol=1e-10)
    config = TrainingConfig(seed=0,device='cpu',training_mode='fixed_steps_v1',steps=2,
        warmup_steps=0,validation_interval_steps=1,batch_size=4,learning_rate=2e-4,
        early_stopping_patience=None,native_preconditioning=(
            'unit_rms_gaussian_tail_image_v1' if tail_order==2 else
            'unit_rms_gaussian_image_v1' if prior=='unit_gaussian' else 'unit_rms_pixel_mixture_image_v1'),
        image_shape=(4,4,3),image_width=4)
    config = replace(config,**({'sigma_min':1/256,'sigma_max':64} if family == 'gaussian_diffusion'
                               else {'time_min':1/65,'time_max':256/257}))
    rng = np.random.default_rng(114)
    import json
    assert TrainingConfig.from_mapping(json.loads(json.dumps(config.to_dict())))==config
    fit,hold = rng.normal(size=(16,4,4,3)),rng.normal(size=(4,4,4,3))
    result = train_model(family,fit,hold,config,tmp_path/'image.pt')
    loaded = load_checkpoint(tmp_path/'image.pt')
    for k,v in result.model.state_dict().items():
        torch.testing.assert_close(v,loaded.model.state_dict()[k],rtol=0,atol=0)
    left = predict_lid(result,hold,.5,divergence_backend='exact')
    right = predict_lid(loaded,hold,.5,divergence_backend='exact')
    np.testing.assert_array_equal(left,right)


@pytest.mark.parametrize('family',['gaussian_diffusion','rectified_flow'])
def test_tail_response_order_in_a_known_linear_residual(family):
    from models.image_gaussian_fields import ImageFieldConfig,NativeGaussianImageField
    class LinearResidual(torch.nn.Module):
        def forward(self,inputs,condition):
            return 2*inputs+.3
    points=torch.randn(1,48,dtype=torch.float64)
    ratios=[]
    for order in (1,2):
        model=NativeGaussianImageField(ImageFieldConfig(48,(4,4,3),4,False,'unit_gaussian',order),native_family=family).double()
        model.field=LinearResidual()
        values=[]
        for lam in (1000.,2000.):
            def posterior(x):
                if family=='gaussian_diffusion':
                    return model(x,torch.tensor(lam,dtype=torch.float64))
                t=1/(1+lam)
                y=t*x
                return y+(1-t)*model(y,torch.tensor(t,dtype=torch.float64))
            jac=torch.autograd.functional.jacobian(lambda x:posterior(x[None])[0],points[0])
            response=float(jac.diagonal().sum())
            residual=2*lam/(1+lam**2)**(1 if order==1 else 1.5)
            expected=48*(1/(1+lam**2)+residual)
            assert response==pytest.approx(expected,rel=1e-9,abs=1e-12)
            values.append(response)
        ratios.append(values[1]/values[0])
    assert ratios[0]==pytest.approx(.5,abs=.001)
    assert ratios[1]==pytest.approx(.25,abs=.001)
