"""Exact mixed radial measure of the released PFGM++ Beta-clamp sampler.

The continuous density is only one component: clipping adds two uniform
spherical measures. It must never be presented as a normalized ordinary PDF.
All scales here are native sigma, before conversion to coordinate RMS.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import betainc, betaincc, betaln, gammaln


@dataclass(frozen=True)
class ClippedPFGMKernel:
    dimension: int
    augmented_dimension: float = 128.
    clip: float = .001
    epsilon: float = 1e-8

    def __post_init__(self):
        if isinstance(self.dimension,bool) or not isinstance(self.dimension,int) or self.dimension<1:
            raise ValueError('dimension must be a positive integer')
        if not math.isfinite(self.augmented_dimension) or self.augmented_dimension<=0:
            raise ValueError('augmented dimension must be positive and finite')
        if not 0<self.clip<.5 or not math.isfinite(self.epsilon) or self.epsilon<0:
            raise ValueError('invalid clipping or numerical epsilon')

    @property
    def beta_parameters(self):
        return self.dimension/2,self.augmented_dimension/2

    def radius(self,beta_value):
        u=np.asarray(beta_value,dtype=float)
        return np.sqrt(self.augmented_dimension)*np.sqrt(u/(1-u+self.epsilon)+self.epsilon)

    @property
    def radius_bounds(self):
        return float(self.radius(self.clip)),float(self.radius(1-self.clip))

    @property
    def atom_masses(self):
        a,b=self.beta_parameters
        return float(betainc(a,b,self.clip)),float(betaincc(a,b,1-self.clip))

    def _beta_density(self,u):
        a,b=self.beta_parameters
        return math.exp((a-1)*math.log(u)+(b-1)*math.log1p(-u)-betaln(a,b))

    def expect_radial(self,function,*,sigma=1.,epsabs=1e-11,epsrel=1e-11):
        """Integrate a scalar/vector radial function, including both atoms."""
        if not math.isfinite(sigma) or sigma<=0:
            raise ValueError('native sigma must be positive and finite')
        a,b=self.beta_parameters
        lo,hi=self.radius_bounds;ml,mh=self.atom_masses
        points=[a/(a+b)]
        if a>1 and b>1:points.append((a-1)/(a+b-2))
        points=sorted({v for v in points if self.clip<v<1-self.clip})
        interior,_=quad_vec(lambda u:np.asarray(function(sigma*self.radius(u)))*self._beta_density(u),
                            self.clip,1-self.clip,points=points,epsabs=epsabs,epsrel=epsrel,limit=200)
        return interior+ml*np.asarray(function(sigma*lo))+mh*np.asarray(function(sigma*hi))

    def dilation_action(self,radial_derivative,*,sigma=1.):
        """Weak log-sigma derivative: E[R f'(R)], including boundary masses.

        This acts on a differentiable radial test function; it is not the
        marginal score of an unknown data distribution or a denoiser readout.
        """
        return self.expect_radial(lambda r:r*np.asarray(radial_derivative(r)),sigma=sigma)

    def rms_factor(self):
        return math.sqrt(float(self.expect_radial(lambda r:r*r))/self.dimension)

    def radial_cdf(self,radius,*,sigma=1.):
        if not math.isfinite(sigma) or sigma<=0:
            raise ValueError('native sigma must be positive and finite')
        r=np.asarray(radius,dtype=float)/sigma
        lo,hi=self.radius_bounds
        v=np.maximum(r*r/self.augmented_dimension-self.epsilon,0.)
        u=np.clip((1+self.epsilon)*v/(1+v),0.,1.)
        result=betainc(*self.beta_parameters,u)
        return np.where(r<lo,0.,np.where(r>=hi,1.,result))

    def log_continuous_density(self,noise,*,sigma=1.):
        """Lebesgue density of the continuous component, not a full log_prob.

        Its integral is 1-minus-the-two-atom-masses. Each boundary sphere
        separately carries its recorded probability mass. Values on the
        spheres are set to -inf; pointwise density there cannot encode atoms.
        """
        if not math.isfinite(sigma) or sigma<=0:
            raise ValueError('native sigma must be positive and finite')
        z=np.asarray(noise,dtype=float)
        if z.ndim<1 or z.shape[-1]!=self.dimension or not np.isfinite(z).all():
            raise ValueError('finite full-dimensional noise vectors are required')
        radius=np.linalg.norm(z,axis=-1)/sigma
        lo,hi=self.radius_bounds
        valid=(radius>lo)&(radius<hi)
        safe=np.where(valid,radius,(lo+hi)/2)
        v=safe*safe/self.augmented_dimension-self.epsilon
        u=(1+self.epsilon)*v/(1+v)
        a,b=self.beta_parameters
        sphere_log_area=math.log(2)+self.dimension/2*math.log(math.pi)-gammaln(self.dimension/2)
        value=((a-1)*np.log(u)+(b-1)*np.log1p(-u)-betaln(a,b)
               +math.log(2*(1+self.epsilon)/self.augmented_dimension)-2*np.log1p(v)
               +(2-self.dimension)*np.log(safe)-sphere_log_area-self.dimension*math.log(sigma))
        return np.where(valid,value,-np.inf)

    def record(self):
        return dict(kind='beta_clamp_mixed_radial_v1',dimension=self.dimension,
                    augmented_dimension=self.augmented_dimension,clip=self.clip,
                    denominator_epsilon=self.epsilon,radicand_epsilon=self.epsilon,
                    unit_sigma_radius_bounds=list(self.radius_bounds),
                    spherical_atom_masses=list(self.atom_masses),
                    coordinate_rms_per_sigma=self.rms_factor(),ordinary_pdf=False)
