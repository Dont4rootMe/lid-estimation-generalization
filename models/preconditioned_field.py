"""Analytic Gaussian input/output scaling for each native vector-field target.

The family objective, sampling law and condition coordinate remain unchanged.
This is a parameterization of the same native output, not a new LID readout.
"""
import torch
from torch import nn

from models.vp_baseline import ConditionedBottleneckMLP, VPBottleneckMLP, VPSchedule
from models.affine_flow import affine_schedule_state, posterior_to_velocity, canonical_schedule


class PreconditionedField(VPBottleneckMLP):
    def __init__(self, architecture, family, training_config):
        ConditionedBottleneckMLP.__init__(self, architecture.ambient_dim,
            architecture.hidden_sizes, architecture.time_dim,
            condition_transform=architecture.condition_transform)
        self.family = family
        self.training_config = training_config
        self.remove_input_skip = training_config.field_preconditioning.endswith('no_input_skip_v1')
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def _final_skip(self, skip):
        if self.remove_input_skip:
            return torch.cat((torch.zeros_like(skip[:, :self.ambient_dim]), skip[:, self.ambient_dim:]), 1)
        return skip

    def channel(self, condition):
        family, cfg = self.family, self.training_config
        if family == 'vp_diffusion':
            schedule = VPSchedule(beta_min=cfg.vp_beta_min, beta_max=cfg.vp_beta_max)
            alpha, beta = schedule.coefficients(condition)
            return (beta / alpha).clamp_min(torch.finfo(condition.dtype).eps), alpha
        if family == 'rectified_flow':
            return (1 - condition) / condition, condition
        if family == 'gaussian_diffusion':
            return condition, torch.ones_like(condition)
        if family == 'brownian_schrodinger_bridge':
            return (condition * cfg.bridge_diffusivity).sqrt(), torch.ones_like(condition)
        scale = condition.exp()
        # Only alpha is needed here. The objective/readout already validates
        # its schedule; constructing all derivatives twice per forward causes
        # repeated CPU/GPU synchronization in concurrent affine-FM training.
        schedule=canonical_schedule(cfg.flow_schedule)
        if schedule=='rectified_linear':alpha=torch.reciprocal(1.0+scale)
        elif schedule=='log_noise':alpha=torch.ones_like(scale)
        else:alpha=torch.rsqrt(1.0+scale.square())
        return scale,alpha

    def posterior(self, inputs, condition):
        condition = torch.as_tensor(condition, dtype=inputs.dtype, device=inputs.device)
        if condition.ndim == 0: condition = condition.expand(len(inputs))
        scale, alpha = self.channel(condition)
        scale, alpha = scale[:, None], alpha[:, None]
        inverse = torch.rsqrt(1 + scale.square())
        canonical = inputs / alpha
        residual = ConditionedBottleneckMLP.forward(self, canonical * inverse, condition)
        return canonical * inverse.square() + scale * inverse * residual

    def forward(self, inputs, condition):
        condition = torch.as_tensor(condition, dtype=inputs.dtype, device=inputs.device)
        if condition.ndim == 0: condition = condition.expand(len(inputs))
        q = self.posterior(inputs, condition)
        if self.family in {'gaussian_diffusion','brownian_schrodinger_bridge'} or (
            self.family=='independent_affine_flow' and self.training_config.flow_parameterization=='posterior_mean'):
            return q
        scale, alpha = self.channel(condition)
        scale, alpha = scale[:, None], alpha[:, None]
        if self.family == 'vp_diffusion':
            prediction=(alpha * q - inputs) / (alpha * scale)
            # A float32 uniform RNG can produce t=0 exactly. At that endpoint
            # epsilon is independent of the observation and its mean is zero.
            return torch.where((condition==0)[:,None],torch.zeros_like(prediction),prediction)
        if self.family == 'rectified_flow':
            return (q - inputs) / (1 - alpha)
        if self.family == 'independent_affine_flow' and self.training_config.flow_parameterization == 'direct_velocity':
            state = affine_schedule_state(scale[:, 0], self.training_config.flow_schedule)
            return posterior_to_velocity(q, inputs, state)
        return q


def build_bottleneck(architecture, family, config):
    if family in {'student_t_flow','pfgmpp'}:
        from models.non_gaussian_fields import build_field
        return build_field(architecture,family,config)
    if config.field_preconditioning == 'image_gaussian_v1':
        from models.shared_image_field import SharedImagePosteriorField
        return SharedImagePosteriorField(architecture,family,config)
    if config.field_preconditioning == 'covariance_span_v1':
        return CovariancePreconditionedField(architecture, family, config)
    if config.field_preconditioning is not None:
        return PreconditionedField(architecture, family, config)
    if config.native_preconditioning == 'unit_rms_gaussian_no_input_skip_v1':
        from models.gaussian_fields import NativeGaussianBottleneck
        return NativeGaussianBottleneck(architecture.ambient_dim,architecture.hidden_sizes,
            architecture.time_dim,condition_transform=architecture.condition_transform,native_family=family)
    if config.posterior_preconditioning is not None:
        from models.vp_baseline import PreconditionedLogNoiseMLP,PreconditionedLogNoiseNoInputSkipMLP
        cls=PreconditionedLogNoiseMLP if config.posterior_preconditioning=='unit_rms_gaussian_v1' else PreconditionedLogNoiseNoInputSkipMLP
        return cls(architecture.ambient_dim,architecture.hidden_sizes,architecture.time_dim,
            condition_transform=architecture.condition_transform)
    if family == 'vp_diffusion':
        return VPBottleneckMLP(architecture.ambient_dim, architecture.hidden_sizes, architecture.time_dim)
    return ConditionedBottleneckMLP(architecture.ambient_dim, architecture.hidden_sizes,
        architecture.time_dim, condition_transform=architecture.condition_transform)


class NoInputSkipCore(ConditionedBottleneckMLP):
    def _final_skip(self, skip):
        return torch.cat((torch.zeros_like(skip[:, :self.ambient_dim]), skip[:, self.ambient_dim:]), 1)


class CovariancePreconditionedField(PreconditionedField):
    """One training-only covariance rule, shared by all vector-field families.

    The Gaussian convolution factors exactly outside an affine data span.
    A fixed relative spectral threshold (1e-8) detects numerical null directions;
    the nonzero principal coordinates use their own Gaussian preconditioners.
    Lambda remains in the ORIGINAL ambient fit-RMS units, without rescaling.
    """
    def __init__(self, architecture, family, training_config):
        nn.Module.__init__(self)
        self.config=architecture;self.ambient_dim=architecture.ambient_dim
        self.family=family;self.training_config=training_config
        rank=training_config.field_projection_rank
        if rank is None or not 0 < rank <= self.ambient_dim:raise ValueError('resolved covariance rank required')
        self.register_buffer('basis',torch.zeros(self.ambient_dim,rank))
        self.register_buffer('variances',torch.ones(rank))
        self.register_buffer('omitted_variance_fraction',torch.zeros(1))
        if training_config.field_backbone == 'spectral_residual_v1':
            from models.spectral_residual_field import SpectralResidualCore
            self.core=SpectralResidualCore(rank,width=training_config.field_residual_width)
        else:
            self.core=NoInputSkipCore(rank,architecture.hidden_sizes,architecture.time_dim,
                                     condition_transform=architecture.condition_transform)
            nn.init.zeros_(self.core.decoder[-1].weight);nn.init.zeros_(self.core.decoder[-1].bias)

    def posterior(self, inputs, condition):
        condition=torch.as_tensor(condition,dtype=inputs.dtype,device=inputs.device)
        if condition.ndim==0:condition=condition.expand(len(inputs))
        scale,alpha=self.channel(condition);scale=scale[:,None];alpha=alpha[:,None]
        coordinate=(inputs/alpha)@self.basis
        inverse=torch.rsqrt(self.variances+scale.square())
        if self.training_config.field_backbone == 'spectral_residual_v1':
            residual=self.core(coordinate*inverse,scale[:,0].log(),scale*inverse)
        else:
            residual=self.core(coordinate*inverse,condition)
        gain=self.variances*inverse.square()
        if self.training_config.field_residual_scaling == 'data_v1':
            # A singular manifold posterior has O(1) normal restoration
            # derivatives as noise vanishes. This factor does not force the
            # residual network's derivative to grow like 1 / noise.
            output_factor=self.variances.sqrt()*gain
        elif self.training_config.field_residual_scaling == 'gaussian_tail_v1':
            # Keep the small-noise residual factor, while making a bounded
            # residual-input Jacobian contribute O(lambda^-2) at large noise.
            output_factor=scale*gain
        else:
            output_factor=scale*self.variances.sqrt()*inverse
        mean=gain*coordinate+output_factor*residual
        return mean@self.basis.T

    def canonical_posterior(self, canonical, noise_ratio):
        """Posterior in X+lambda Z coordinates, before native target conversion."""
        scale=canonical.new_full((len(canonical),),float(noise_ratio))
        if self.family=='vp_diffusion':
            schedule=VPSchedule(self.training_config.vp_beta_min,self.training_config.vp_beta_max)
            condition=canonical.new_full((len(canonical),),schedule.time_for_lambda(float(noise_ratio)))
            alpha,_=schedule.coefficients(condition)
        elif self.family=='rectified_flow':
            alpha=1/(1+scale);condition=alpha
        elif self.family=='gaussian_diffusion':
            alpha=torch.ones_like(scale);condition=scale
        elif self.family=='brownian_schrodinger_bridge':
            alpha=torch.ones_like(scale);condition=scale.square()/self.training_config.bridge_diffusivity
        else:
            alpha=affine_schedule_state(scale,self.training_config.flow_schedule).alpha
            condition=scale.log()
        return self.posterior(alpha[:,None]*canonical,condition)


def covariance_span_value_and_trace(model, canonical, noise_ratio):
    """Exact ambient posterior trace using the declared learned affine span.

    This uses its global affine rank, not an intrinsic dimension or LID label.
    Dual output directions account for finite-precision nonorthogonality of
    stored basis columns. The nonlinear Jacobian is still differentiated.
    """
    if not isinstance(model,CovariancePreconditionedField):
        raise TypeError('an explicit covariance-span posterior model is required')
    with torch.enable_grad():
        x=canonical.detach().requires_grad_(True)
        value=model.canonical_posterior(x,noise_ratio)
        basis=model.basis.to(dtype=x.dtype,device=x.device)
        gram=basis.double().T@basis.double()
        dual=(basis.double()@torch.linalg.inv(gram)).to(dtype=x.dtype)
        response=torch.zeros(len(x),dtype=x.dtype,device=x.device)
        for j in range(basis.shape[1]):
            derivative=torch.autograd.grad(value,x,grad_outputs=dual[:,j].expand_as(x),
                retain_graph=j+1<basis.shape[1])[0]
            response=response+(derivative*basis[:,j]).sum(1)
        return value.detach(),response.detach()


def training_covariance_span(normalized_train,device,minimum_rank=1):
    """Fit once from optimizer-fit observations only; no label/holdout access."""
    with torch.no_grad():
        x=normalized_train.to(device=device,dtype=torch.float64)
        covariance=x.T@x/len(x)
        eigenvalues,eigenvectors=torch.linalg.eigh(covariance)
        keep=eigenvalues>eigenvalues[-1]*1e-8
        # Treat a clear numerical nullspace as analytic Gaussian noise. Do not
        # silently discard a genuine weak coordinate near the coarse cutoff.
        # The benchmark spans have discarded ratios far below this guard.
        if bool((~keep).any()) and float(eigenvalues[~keep].max()/eigenvalues[-1])>1e-12:
            # Float32 native centering can leave a tiny constant normal offset.
            # A mean offset is not an additional varying data coordinate.
            mean64=x.mean(0)
            centered_values=torch.linalg.eigvalsh(covariance-mean64[:,None]*mean64[None,:])
            if int((centered_values>centered_values[-1]*1e-12).sum())>int(keep.sum()):
                keep=eigenvalues>eigenvalues[-1]*1e-12
        if not bool(keep.any()):raise ValueError('zero-variance training data')
        keep[-minimum_rank:]=True
        variance=eigenvalues[keep].clamp_min(0);basis=eigenvectors[:,keep]
        omitted=eigenvalues[~keep].clamp_min(0).sum()/eigenvalues.clamp_min(0).sum()
        return basis.float(),variance.float(),omitted.float()
