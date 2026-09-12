"""Mixed-kernel identities; no learned model or optimizer updates."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betainc,betaln,gammaln

from models.pfgm_kernel import ClippedPFGMKernel


@pytest.mark.parametrize('dimension',[1,3,30,784,3072])
def test_mixed_measure_mass_and_rms_match_independent_beta_integration(dimension):
    k=ClippedPFGMKernel(dimension)
    assert abs(float(k.expect_radial(lambda r:1.))-1)<3e-11
    a,b=dimension/2,64.;l,h=.001,.999
    f=lambda u:u/(1-u+1e-8)+1e-8
    integral=quad(lambda u:f(u)*math.exp((a-1)*math.log(u)+(b-1)*math.log1p(-u)-betaln(a,b)),
                  l,h,points=[a/(a+b)],epsabs=1e-10,epsrel=1e-10,limit=200)[0]
    expected=math.sqrt(128*(integral+f(l)*betainc(a,b,l)+f(h)*(1-betainc(a,b,h)))/dimension)
    assert math.isclose(k.rms_factor(),expected,rel_tol=2e-10,abs_tol=2e-12)
    lo,hi=k.radius_bounds;ml,mh=k.atom_masses
    assert float(k.radial_cdf(lo))==pytest.approx(ml,abs=1e-14)
    assert k.radial_cdf(lo*.999)==0 and k.radial_cdf(hi)==1


@pytest.mark.parametrize('dimension',[1,3,30])
def test_continuous_density_keeps_missing_atomic_mass_explicit(dimension):
    k=ClippedPFGMKernel(dimension)
    lo,hi=k.radius_bounds;area=2*math.pi**(dimension/2)/math.exp(gammaln(dimension/2))
    direction=np.zeros(dimension);direction[0]=1
    mass=quad(lambda r:float(np.exp(k.log_continuous_density(r*direction)))*area*r**(dimension-1),
              lo,hi,points=[float(k.radius(dimension/(dimension+128)))],epsabs=1e-10,limit=200)[0]
    assert abs(mass+sum(k.atom_masses)-1)<2e-10
    assert np.isneginf(k.log_continuous_density(lo*.99*direction))
    assert np.isneginf(k.log_continuous_density(hi*1.01*direction))


def test_weak_scale_derivative_includes_shell_motion():
    k=ClippedPFGMKernel(1)
    sigma=.7;h=1e-5
    f=lambda r:np.exp(-r*r/2)
    derivative=k.dilation_action(lambda r:-r*np.exp(-r*r/2),sigma=sigma)
    finite=(k.expect_radial(f,sigma=sigma*math.exp(h))-k.expect_radial(f,sigma=sigma*math.exp(-h)))/(2*h)
    assert abs(float(derivative-finite))<1e-9
    # In this valid low-dimensional case the lower shell is appreciable.
    assert k.atom_masses[0]>.1
