"""t-Flow and PFGM++ with unit-RMS evaluation scales and native response LID.

t-Flow: Pandey et al., arXiv:2410.14171v2, Appendix B (164--166).
PFGM++: Xu et al., PMLR 202 (2023), equations 4--6 and the EDM loss.
Both use a radial Student-t kernel, not independent coordinatewise Student-t.
No Gaussian posterior-to-score or squared-displacement correction is used.
"""
import math

import torch

from models.noise_pairing import paired_corruption
from models.preconditioned_field import CovariancePreconditionedField
from models.shared_image_field import SharedImagePosteriorField
from models.ambient_field import AmbientPosteriorField


FAMILIES = frozenset({'student_t_flow', 'pfgmpp'})


def validate_config(family, config):
    if family not in FAMILIES:
        raise ValueError('unknown non-Gaussian family')
    df = config.kernel_df
    if isinstance(df, bool) or not isinstance(df, int) or df <= 4:
        raise ValueError('this finite-variance implementation requires integer kernel_df > 4')
    if (config.sigma_min is None or config.sigma_max is None or
            not 0 < config.sigma_min < config.sigma_max < math.inf):
        raise ValueError('explicit positive common RMS scale support required')
    if config.field_preconditioning not in {'covariance_span_v1', 'ambient_isotropic_v1', 'image_gaussian_v1'}:
        raise ValueError('non-Gaussian models require the shared routed field backbone')
    expected = 'image_unet_v1' if config.field_preconditioning=='image_gaussian_v1' else 'spectral_residual_v1'
    if config.field_backbone != expected:
        raise ValueError('non-Gaussian model backbone must follow the shared representation route')
    if config.training_target != 'sample_v1':
        raise ValueError('Gaussian empirical teachers are invalid for a Student-t kernel')
    if config.kernel_log_scale_std is None or not 0 < config.kernel_log_scale_std < math.inf:
        raise ValueError('positive lognormal scale std required')
    if config.kernel_log_scale_mean is None or not math.isfinite(config.kernel_log_scale_mean):
        raise ValueError('finite lognormal scale mean required')


def contract(family, config):
    validate_config(family, config)
    return dict(schema_version=1, family=family, kernel='radial_multivariate_student_t',
        kernel_df=config.kernel_df, evaluation_scale='per_coordinate_rms_noise',
        rms_to_student_scale=math.sqrt((config.kernel_df-2)/config.kernel_df),
        rms_support=[config.sigma_min, config.sigma_max], primary_readout='response',
        native_field='dataward_rectified_velocity' if family=='student_t_flow' else 'radial_poisson_field',
        objective='native_student_noise_mse' if family=='student_t_flow' else 'edm_weighted_posterior_mse',
        scale_sampling='uniform_native_time' if family=='student_t_flow' else 'truncated_native_sigma_lognormal',
        pfgm_log_sigma_mean=config.kernel_log_scale_mean if family=='pfgmpp' else None,
        pfgm_log_sigma_std=config.kernel_log_scale_std if family=='pfgmpp' else None,
        pfgm_sigma_data=1. if family=='pfgmpp' else None,
        normal_conditioning=('exact_student_conditional_scale_v1'
            if config.field_preconditioning=='covariance_span_v1' else 'full_ambient_learned_input_output'),
        gaussian_full_or_score_conversion=False)


def student_unit_rms(shape, df, *, device, dtype, generator):
    """One shared chi-square denominator per ambient vector; no tail clipping."""
    gaussian = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    chi = torch.randn((shape[0], df), device=device, dtype=dtype, generator=generator).square().sum(1)
    denominator = torch.sqrt(chi / (df-2))
    return gaussian / denominator.reshape(-1, *([1]*(len(shape)-1)))


def sample_rms_scale(n, family, config, *, device, dtype, generator):
    lo, hi = config.sigma_min, config.sigma_max
    k = math.sqrt(config.kernel_df/(config.kernel_df-2))
    # Work in float64 for inverse-CDF tails, then use the shared float32 model.
    u = torch.rand(n, device=device, dtype=torch.float64, generator=generator)
    if family == 'student_t_flow':
        t = k/(k+hi) + u*(k/(k+lo)-k/(k+hi))
        scale = k*(1-t)/t
    else:
        mu, sd = config.kernel_log_scale_mean, config.kernel_log_scale_std
        low_cdf = .5*(1+math.erf((math.log(lo/k)-mu)/sd/math.sqrt(2)))
        high_cdf = .5*(1+math.erf((math.log(hi/k)-mu)/sd/math.sqrt(2)))
        normal = math.sqrt(2)*torch.erfinv(2*(low_cdf+u*(high_cdf-low_cdf))-1)
        scale = k*torch.exp(mu+sd*normal)
    return scale.to(dtype=dtype)


def conditional_rms_scale(canonical, basis, scale, df):
    """Exact radial-t conditioning on the observed component normal to a span.

    With ambient N, span rank r and normal observation n, the active kernel has
    df'=df+N-r and scale variance ((df-2)*lambda_RMS^2+||n||^2)/(df'-2).
    The posterior therefore needs no extra learnable normal-coordinate input.
    """
    active = canonical @ basis
    normal = canonical - active @ basis.T
    numerator = (df-2)*scale.square() + normal.square().sum(1)
    return torch.sqrt(numerator/(df+canonical.shape[1]-basis.shape[1]-2))


class NativeStudentInterface:
    def canonical_posterior(self, canonical, noise_ratio):
        scale = torch.as_tensor(noise_ratio, device=canonical.device, dtype=canonical.dtype)
        if scale.ndim == 0: scale = scale.expand(len(canonical))
        return self.posterior(canonical, scale)

    def native_velocity(self, state, time):
        time = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        if time.ndim == 0: time = time.expand(len(state))
        scale = math.sqrt(self.training_config.kernel_df/(self.training_config.kernel_df-2))*(1-time)/time
        return (self.posterior(state/time[:, None], scale)-state)/(1-time[:, None])

    def poisson_radial_field(self, canonical, radius):
        radius = torch.as_tensor(radius, dtype=canonical.dtype, device=canonical.device)
        if radius.ndim == 0: radius = radius.expand(len(canonical))
        scale = radius/math.sqrt(self.training_config.kernel_df-2)
        return (canonical-self.posterior(canonical, scale))/radius[:, None]

    def forward(self, inputs, rms_scale):
        return self.posterior(inputs, rms_scale)


class StudentCovarianceField(NativeStudentInterface, CovariancePreconditionedField):
    def posterior(self, inputs, condition):
        scale = torch.as_tensor(condition, dtype=inputs.dtype, device=inputs.device)
        if scale.ndim == 0: scale = scale.expand(len(inputs))
        effective = conditional_rms_scale(inputs, self.basis, scale, self.training_config.kernel_df)
        # The same parameter count and spectral core as every Gaussian field.
        # These are neural scaling coefficients, not a Gaussian posterior claim.
        coordinate = inputs @ self.basis
        inverse = torch.rsqrt(self.variances+effective[:, None].square())
        gain = self.variances*inverse.square()
        residual = self.core(coordinate*inverse, effective.log(), effective[:, None]*inverse)
        if self.training_config.field_residual_scaling=='data_v1':
            factor = self.variances.sqrt()*gain
        elif self.training_config.field_residual_scaling=='gaussian_tail_v1':
            factor = effective[:, None]*gain
        else:
            factor = effective[:, None]*self.variances.sqrt()*inverse
        return (gain*coordinate+factor*residual) @ self.basis.T


class StudentAmbientField(NativeStudentInterface, AmbientPosteriorField):
    def channel(self, condition):
        return condition, torch.ones_like(condition)


class StudentImageField(NativeStudentInterface, SharedImagePosteriorField):
    def channel(self, condition):
        # Full ambient images retain all normal information in their input.
        return condition, torch.ones_like(condition)


def build_field(architecture, family, config):
    validate_config(family, config)
    cls = {'covariance_span_v1': StudentCovarianceField,
        'ambient_isotropic_v1': StudentAmbientField,
        'image_gaussian_v1': StudentImageField}[config.field_preconditioning]
    return cls(architecture, family, config)


def loss(family, model, clean, config, generator):
    scale = sample_rms_scale(len(clean), family, config, device=clean.device,
        dtype=clean.dtype, generator=generator)
    noise = student_unit_rms(clean.shape, config.kernel_df, device=clean.device,
        dtype=clean.dtype, generator=generator)
    clean, scale, noise = paired_corruption(model, clean, scale, noise)
    prediction = model(clean+scale[:, None]*noise, scale)
    sigma = scale*math.sqrt((config.kernel_df-2)/config.kernel_df)
    weight = sigma.reciprocal().square()
    if family=='pfgmpp': weight = weight+1.  # sigma_data=1 in fit-RMS units.
    return (weight[:, None]*(prediction-clean).square()).mean()
