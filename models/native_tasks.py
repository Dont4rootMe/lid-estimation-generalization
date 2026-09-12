"""Native learning tasks, explicitly separated from frozen LID readouts.

Architecture is shared; inputs, conditions, raw outputs and losses are not
canonicalized. Old checkpoints keep their historical routes. See
docs/native_methods.md for source pins and the explicit FLIPD VE bug fix.
"""
from functools import lru_cache
import math
import numpy as np
import torch
from torch import nn

from models.spectral_residual_field import SpectralResidualCore
from models.point_residual_field import PointResidualCore
from models.image_gaussian_fields import ImageUNet
from models.shared_image_field import to_image, from_image
from models.vp_baseline import VPSchedule

VARIANTS = (
    'vp_diffusion', 've_diffusion', 'rectified_flow', 'scale_conditioned_nf',
    'schrodinger_bridge', 'posterior_rectified_flow',
    'direct_log_noise_affine_flow', 'posterior_log_noise_affine_flow',
    'direct_vp_trigonometric_flow', 'posterior_vp_trigonometric_flow',
    't_flowmatching', 'pfgmpp',
)
CUSTOM = frozenset(('posterior_rectified_flow', 'direct_log_noise_affine_flow',
                   'posterior_log_noise_affine_flow', 'posterior_vp_trigonometric_flow'))
FAMILIES = dict(vp_diffusion='vp_diffusion',ve_diffusion='gaussian_diffusion',
    rectified_flow='rectified_flow',scale_conditioned_nf='scale_conditioned_normalizing_flow',
    schrodinger_bridge='brownian_schrodinger_bridge',t_flowmatching='student_t_flow',pfgmpp='pfgmpp',
    **{v:'independent_affine_flow' for v in CUSTOM|{'direct_vp_trigonometric_flow'}})
SOURCES = {
    'flipd': 'https://github.com/layer6ai-labs/flipd/tree/05ab170c2c9bcada3f9286c3ef86db31b80925fa',
    'rf': 'https://github.com/gnobitab/RectifiedFlow/tree/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6',
    'interflow': 'https://arxiv.org/abs/2209.15571',
    't_flow': 'https://arxiv.org/abs/2410.14171',
    'pfgmpp': 'https://github.com/Newbeeer/pfgmpp/tree/d57c1ee4488d8e4064a5b9c8548792f00395aa8b',
    'dsb': 'https://github.com/JTT94/diffusion_schrodinger_bridge/tree/1c82eba0a16aea3333ac738dde376b12a3f97f21',
    'nf': 'https://arxiv.org/abs/1605.08803',
}


def validate_config(cfg):
    if cfg.native_variant not in VARIANTS:
        raise ValueError('unknown native variant; direct_rectified_flow is retired')
    if cfg.native_variant != 'scale_conditioned_nf':
        if cfg.field_preconditioning != 'native_task_v1':
            raise ValueError('native learning forbids the common posterior wrapper')
        if cfg.field_projection_rank is not None or cfg.training_target != 'sample_v1':
            raise ValueError('native tasks require full inputs and sampled targets')
        if cfg.noise_pairing != 'iid':
            raise ValueError('published native tasks use independent corruption samples')
        if cfg.image_shape is None and cfg.field_backbone not in {'point_residual_v1', 'spectral_residual_v1'}:
            raise ValueError('native vector fields require the point core or an explicit legacy spectral checkpoint')
    if cfg.ve_forward_policy != 'corrected_flipd_ve':
        raise ValueError('VE must use X + sigma * Z with the same Z in the target')
    if not 0 < cfg.pfgm_beta_clip < .5 or cfg.pfgm_sigma_data <= 0:
        raise ValueError('invalid PFGM++ native constants')
    if cfg.native_variant == 'schrodinger_bridge':
        if cfg.dsb_num_steps < 2 or cfg.dsb_step_size <= 0 or cfg.dsb_ipf_rounds < 1:
            raise ValueError('invalid DSB discretization')
        if cfg.steps is None or cfg.steps % (2 * cfg.dsb_ipf_rounds):
            raise ValueError('DSB budget must complete equal forward/backward IPF passes')
        if cfg.ema_decay is not None:
            raise ValueError('DSB retains the final IPF pair, without cross-pass EMA')
        if (cfg.bridge_diffusivity != 2. or
                not math.isclose(cfg.bridge_terminal_time,cfg.dsb_num_steps*cfg.dsb_step_size)):
            raise ValueError('DSB native diffusion variance/time differs from its discretization')


def contract(cfg, family):
    validate_config(cfg)
    v = cfg.native_variant
    if family != FAMILIES[v]:
        raise ValueError('native variant and declared model family disagree')
    raw = ('negative_standard_noise' if v in {'vp_diffusion', 've_diffusion'} else
           'velocity' if v in {'rectified_flow', 'direct_vp_trigonometric_flow'} else
           'log_noise_velocity' if v == 'direct_log_noise_affine_flow' else
           'student_noise' if v == 't_flowmatching' else
           'edm_residual' if v == 'pfgmpp' else
           'two_transition_increments' if v == 'schrodinger_bridge' else
           'conditional_invertible_transform' if v == 'scale_conditioned_nf' else 'clean_endpoint')
    return dict(schema_version=3, family=family, native_variant=v,
        learning_contract='native_tasks_v1', raw_network_output=raw,
        source_status='author_parameterization_control' if v in CUSTOM else
            'sigma_conditioned_adaptation' if v == 'scale_conditioned_nf' else 'published_learning_task',
        common_posterior_training=False, field_preconditioning=cfg.field_preconditioning,
        field_backbone=cfg.field_backbone, sources=SOURCES,
        ve_forward_policy=cfg.ve_forward_policy if v == 've_diffusion' else None,
        clipping=dict(beta_bounds=[cfg.pfgm_beta_clip, 1-cfg.pfgm_beta_clip],
                      denominator_epsilon=1e-8, radicand_epsilon=1e-8) if v == 'pfgmpp' else None,
        full_readout_status='blocked_pending_clipped_kernel_identity' if v == 'pfgmpp' else 'defined',
        dsb=dict(reference='brownian', diffusion_variance=2., source='independent_standard_gaussian',
                 terminal='data', num_steps=cfg.dsb_num_steps, step_size=cfg.dsb_step_size,
                 ipf_rounds=cfg.dsb_ipf_rounds, raw_output='transition_increment',
                 checkpoint_selection='final_ipf_pair') if v == 'schrodinger_bridge' else None,
        settings=cfg.to_dict())


def _condition(value, x):
    value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    if value.ndim == 0:
        value = value.expand(len(x))
    if value.shape != (len(x),):
        raise ValueError('one scalar condition per sample is required')
    return value


class NativeField(nn.Module):
    def __init__(self, architecture, family, cfg):
        super().__init__()
        self.config, self.family, self.training_config = architecture, family, cfg
        self.ambient_dim = architecture.ambient_dim
        if cfg.image_shape:
            self.core = ImageUNet(cfg.image_shape[-1], cfg.image_width)
        elif cfg.field_backbone == 'point_residual_v1':
            self.core = PointResidualCore(self.ambient_dim, cfg.field_residual_width)
        elif cfg.field_backbone == 'spectral_residual_v1':
            # Reconstruct historical weights exactly; current runs use points.
            self.core = SpectralResidualCore(self.ambient_dim, cfg.field_residual_width)
        else:
            raise ValueError('unsupported native vector backbone')

    def raw(self, inputs, condition, core=None):
        condition = _condition(condition, inputs)
        core = self.core if core is None else core
        cfg = self.training_config
        if cfg.image_shape:
            return from_image(core(to_image(inputs, cfg.image_shape, cfg.image_layout), condition), cfg.image_layout)
        if cfg.field_backbone == 'point_residual_v1':
            return core(inputs, condition)
        # Historical native-v9/v10 checkpoints retained unattenuated features.
        return core(inputs, condition, torch.zeros_like(inputs))

    def forward(self, inputs, condition):
        """Raw declared output, except PFGM++'s explicitly native EDM head."""
        if self.training_config.native_variant == 'pfgmpp':
            sigma = _condition(condition, inputs)
            sd = self.training_config.pfgm_sigma_data
            den = sigma.square() + sd**2
            skip = sd**2 / den
            out = sigma * sd / den.sqrt()
            residual = self.raw(inputs / den.sqrt()[:, None], sigma.log()/4)
            return skip[:, None]*inputs + out[:, None]*residual
        return self.raw(inputs, condition)

    def canonical_posterior(self, x, scale):
        """Evaluation only: differentiate through native coordinates at fixed x."""
        cfg, v = self.training_config, self.training_config.native_variant
        lam = _condition(scale, x)
        if v == 'vp_diffusion':
            integral = torch.log1p(lam.square())
            delta = cfg.vp_beta_max-cfg.vp_beta_min
            t = (2*integral)/(cfg.vp_beta_min+torch.sqrt(cfg.vp_beta_min**2+2*delta*integral))
            alpha = torch.rsqrt(1+lam.square())
            return x+lam[:, None]*self(alpha[:, None]*x, t)
        if v == 've_diffusion':
            t = (lam.log()-math.log(cfg.sigma_min))/math.log(cfg.sigma_max/cfg.sigma_min)
            return x+lam[:, None]*self(x, t)
        if v in {'rectified_flow', 'posterior_rectified_flow'}:
            t = 1/(1+lam)
            if v == 'posterior_rectified_flow':
                return self(t[:, None]*x, lam.log())
            return t[:, None]*x+(1-t)[:, None]*self(t[:, None]*x, 999*t)
        if v == 'direct_log_noise_affine_flow':
            return x-self(x, lam.log())
        if v == 'posterior_log_noise_affine_flow':
            return self(x, lam.log())
        if v in {'direct_vp_trigonometric_flow', 'posterior_vp_trigonometric_flow'}:
            a = torch.rsqrt(1+lam.square()); b = lam*a
            if v.startswith('posterior_'):
                return self(a[:, None]*x, lam.log())
            t = 2/math.pi*torch.atan(lam.reciprocal())
            return a.square()[:, None]*x+(2/math.pi*b)[:, None]*self(a[:, None]*x, t)
        if v == 't_flowmatching':
            k = math.sqrt(cfg.kernel_df/(cfg.kernel_df-2))
            s = lam/k; t = 1/(1+s)
            return x-s[:, None]*self(t[:, None]*x, 1-t)
        if v == 'pfgmpp':
            return self(x, lam/pfgm_rms_factor(self.ambient_dim, cfg.kernel_df, cfg.pfgm_beta_clip))
        raise ValueError('unsupported native posterior adapter')


def build_field(architecture, family, cfg):
    validate_config(cfg)
    if family != FAMILIES[cfg.native_variant]:
        raise ValueError('native variant and declared model family disagree')
    if cfg.native_variant == 'schrodinger_bridge':
        from models.native_bridge import NativeDSB
        return NativeDSB(architecture, family, cfg)
    return NativeField(architecture, family, cfg)


@lru_cache(maxsize=None)
def pfgm_rms_factor(dimension, df, clip=.001):
    """Actual RMS/native-sigma ratio for the author's clamped Beta sampler.

    Clipping changes the second moment; using sqrt(df/(df-2)) would silently
    mislabel the common physical noise grid. Quadrature is deterministic.
    """
    from models.pfgm_kernel import ClippedPFGMKernel
    return ClippedPFGMKernel(dimension,df,clip).rms_factor()


def pfgm_noise(clean, sigma, df, clip, generator):
    """Author sampler: Beta clamp, both 1e-8 terms, uniform angular direction."""
    n, dimension = clean.shape
    device = clean.device
    a = torch.full((n,), dimension/2, device=device, dtype=torch.float64)
    b = torch.full((n,), df/2, device=device, dtype=torch.float64)
    ga = torch._standard_gamma(a, generator=generator)
    gb = torch._standard_gamma(b, generator=generator)
    u = (ga/(ga+gb)).clamp(clip,1-clip)
    radius = sigma.double()*math.sqrt(df)*torch.sqrt(u/(1-u+1e-8)+1e-8)
    direction = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
    direction = direction/direction.norm(dim=1,keepdim=True)
    return (direction.double()*radius[:,None]).to(clean.dtype)


def sample_batch(clean, cfg, generator):
    """Return native inputs, network condition, raw target and per-row weight."""
    n = len(clean); device, dtype = clean.device, clean.dtype
    rand = lambda: torch.rand(n,device=device,dtype=dtype,generator=generator)
    z = torch.randn(clean.shape,device=device,dtype=dtype,generator=generator)
    v = cfg.native_variant
    w = torch.ones(n,device=device,dtype=dtype)
    if v in {'vp_diffusion','ve_diffusion'}:
        t = rand()
        if v == 'vp_diffusion':
            alpha,beta = VPSchedule(cfg.vp_beta_min,cfg.vp_beta_max).coefficients(t)
        else:
            alpha = torch.ones_like(t)
            beta = cfg.sigma_min*(cfg.sigma_max/cfg.sigma_min)**t
        return alpha[:,None]*clean+beta[:,None]*z, t, -z, w
    if v == 'rectified_flow':
        t = cfg.time_min+(cfg.time_max-cfg.time_min)*rand()
        return t[:,None]*clean+(1-t)[:,None]*z,999*t,clean-z,w
    if v == 'direct_vp_trigonometric_flow':
        t = rand(); a = torch.sin(math.pi/2*t); b = torch.cos(math.pi/2*t)
        return a[:,None]*clean+b[:,None]*z,t,math.pi/2*(b[:,None]*clean-a[:,None]*z),w
    if v in CUSTOM:
        lam = torch.exp(math.log(cfg.flow_noise_ratio_min)+rand()*math.log(cfg.flow_noise_ratio_max/cfg.flow_noise_ratio_min))
        a = (1/(1+lam) if cfg.flow_schedule == 'rectified_linear' else
             torch.rsqrt(1+lam.square()) if cfg.flow_schedule == 'vp_trigonometric' else torch.ones_like(lam))
        y = a[:,None]*(clean+lam[:,None]*z)
        target = lam[:,None]*z if v == 'direct_log_noise_affine_flow' else clean
        return y,lam.log(),target,lam.reciprocal().square()
    if v == 't_flowmatching':
        t = rand()
        chi = torch.randn((n,cfg.kernel_df),device=device,dtype=dtype,generator=generator).square().sum(1)
        eta = z/torch.sqrt(chi/cfg.kernel_df)[:,None]
        return t[:,None]*clean+(1-t)[:,None]*eta,1-t,eta,w
    if v == 'pfgmpp':
        sigma = torch.exp(cfg.kernel_log_scale_mean+cfg.kernel_log_scale_std*
                          torch.randn(n,device=device,dtype=dtype,generator=generator))
        noise = pfgm_noise(clean,sigma,cfg.kernel_df,cfg.pfgm_beta_clip,generator)
        weight = (sigma.square()+cfg.pfgm_sigma_data**2)/(sigma*cfg.pfgm_sigma_data).square()
        return clean+noise,sigma,clean,weight
    raise ValueError('this variant has no independent-corruption regression')


def objective(model, clean, cfg, generator):
    if cfg.native_variant == 'schrodinger_bridge':
        return model.ipf_loss(clean,generator)
    inputs,condition,target,weight = sample_batch(clean,cfg,generator)
    error = (model(inputs,condition)-target).square()
    # FLIPD explicitly sums coordinates; the other recipes use mean MSE.
    rows = error.sum(1) if cfg.native_variant in {'vp_diffusion','ve_diffusion'} else error.mean(1)
    return (rows*weight).mean()


def scale_support(cfg):
    v = cfg.native_variant
    if v == 've_diffusion': return cfg.sigma_min,cfg.sigma_max
    if v == 'vp_diffusion':
        return 0.,math.sqrt(math.expm1((cfg.vp_beta_min+cfg.vp_beta_max)/2))
    if v == 'rectified_flow': return 0.,(1-cfg.time_min)/cfg.time_min
    if v == 'scale_conditioned_nf': return cfg.epsilon_min,cfg.epsilon_max
    if v in CUSTOM: return cfg.flow_noise_ratio_min,cfg.flow_noise_ratio_max
    if v == 'schrodinger_bridge':
        return math.sqrt(2*cfg.dsb_step_size),math.sqrt(2*cfg.dsb_step_size*cfg.dsb_num_steps)
    return 0.,math.inf


def valid_scales(cfg, scales, *, nf_stencil=False):
    values = np.asarray(scales,dtype=float)
    lo,hi = scale_support(cfg)
    margin = math.exp(.1) if nf_stencil else 1.
    return np.isfinite(values)&(values>0)&(values/margin>=lo)&(values*margin<=hi)


class PosteriorReadout(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x, scale):
        return self.model.canonical_posterior(x,scale)


def predict(model, query, scale, mean, normalization, readout, backend, probes, seed, batch_size):
    from models.neural_fields import exact_divergence, hutchinson_divergence
    from models.student_density import density_dilation
    cfg = model.training_config
    if readout not in {'full','response','fm_to_score'}:
        raise ValueError('unknown native field readout')
    if not valid_scales(cfg,[scale])[0]:
        raise ValueError('evaluation scale outside the native training/discretization support')
    if cfg.native_variant == 'pfgmpp' and readout != 'response':
        raise NotImplementedError('PFGM++ author clipping is preserved; Full readout requires a clipped-kernel identity. The untruncated Student Full is intentionally not substituted.')
    if backend not in {'exact','hutchinson'} or batch_size < 1:
        raise ValueError('invalid native derivative backend or batch size')
    from models.training import _flat_finite_data
    data = _flat_finite_data(query,name='query')
    if data.shape[1] != model.ambient_dim:
        raise ValueError('query ambient dimension does not match the native model')
    data = (data-mean.reshape(1,-1))/normalization
    parameter = next(model.parameters()); model.eval()
    adapter = PosteriorReadout(model)
    generator = torch.Generator(device=parameter.device).manual_seed(seed)
    result = []
    for start in range(0,len(data),batch_size):
        x = data[start:start+batch_size].to(device=parameter.device,dtype=parameter.dtype)
        response = (exact_divergence(adapter,x,scale) if backend == 'exact' else
                    hutchinson_divergence(adapter,x,scale,num_probes=probes,seed=None,generator=generator))
        if readout == 'response':
            value = response
        elif cfg.native_variant == 't_flowmatching':
            value = density_dilation(adapter,x,scale,response,cfg.kernel_df)[0]
        else:
            with torch.no_grad():
                q = adapter(x,scale)
                value = response+((q-x)/scale).square().sum(1)
        result.append(value.detach().double().cpu())
    values = torch.cat(result).numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError('native readout produced non-finite values')
    return values
