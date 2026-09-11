"""Continuous Student-t posterior via a normalized Gamma/Gaussian integral.

For a=(nu-2)*lambda_raw_RMS**2 and p=(nu+ambient)/2,
(1+d_squared/a)**(-p) = E[exp(-U*d_squared/a)], U~Gamma(p,1).
Gaussian conditional means, covariance traces and evidence therefore give both
the Student posterior and its derivative, including the mixture-weight term.
These references are evaluation only and are never training targets.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import logsumexp, roots_legendre

from experiments.lambda_repair_reference import spiral_posterior


@dataclass
class Reference:
    mean: np.ndarray
    response: float
    log_mass: float
    metadata: dict


@lru_cache(maxsize=32)
def gamma_rule(shape,order):
    """Normalized Golub-Welsch weights avoid Gamma(shape) overflow at N=784."""
    index=np.arange(order,dtype=float)
    diagonal=2*index+shape
    off=np.sqrt(np.arange(1,order)*(np.arange(1,order)+shape-1))
    nodes,vectors=eigh_tridiagonal(diagonal,off)
    weights=vectors[0]**2
    return nodes,weights/weights.sum()


@lru_cache(maxsize=12)
def legendre(order):return roots_legendre(order)


def gaussian_exp(query,sigma,*,metric,order=64,cutoff=10.):
    """Actual uniform-u, uniform-angle Exp law; use the full renderer Gram.

    Axial and angular integration are localized with Euclidean lower bounds
    using the smallest metric eigenvalue, then integrated by product quadrature.
    Radius varies as 3*exp(-u). No point-cloud KDE or radius-frozen cylinder.
    """
    q=np.asarray(query,dtype=float);g=np.asarray(metric,dtype=float)
    bound=sigma/math.sqrt(float(np.linalg.eigvalsh(g).min()))
    lo=max(0.,q[0]+4-cutoff*bound);hi=min(8.,q[0]+4+cutoff*bound)
    if hi<=lo:raise ValueError('Exp reference query is outside the localized support')
    nodes,weights=legendre(order)
    u=(hi+lo)/2+(hi-lo)/2*nodes
    radius=3*np.exp(-u);rho=np.linalg.norm(q[1:])
    angle=math.atan2(q[1],q[2])
    width=(2*np.arcsin(np.minimum(1.,cutoff*bound/(2*np.sqrt(radius*rho))))
        if rho else np.full_like(radius,math.pi))
    theta=angle+width[:,None]*nodes
    points=np.stack((np.broadcast_to(u[:,None]-4,theta.shape),
        radius[:,None]*np.sin(theta),radius[:,None]*np.cos(theta)),axis=-1).reshape(-1,3)
    difference=points-q
    squared=np.einsum('ni,ij,nj->n',difference,g,difference)
    quadrature=((hi-lo)/2*weights[:,None]*width[:,None]*weights[None,:]).reshape(-1)/(8*2*math.pi)
    logits=np.log(quadrature)-squared/(2*sigma*sigma)
    log_mass=logsumexp(logits);probability=np.exp(logits-log_mass)
    shift=probability@difference;centered=difference-shift
    covariance=(centered*probability[:,None]).T@centered
    return Reference(q+shift,float(np.trace(g@covariance)/sigma**2),float(log_mass),
        dict(order=order,cutoff=cutoff,nodes=len(points)))


def gaussian_reference(dataset,query,sigma,*,metric,order=64,cutoff=10.):
    if dataset=='e6_exp_pca':
        return gaussian_exp(query,sigma,metric=metric,order=order,cutoff=cutoff)
    if dataset=='e1_spiral_pca':
        result=spiral_posterior(query,sigma,metric=metric,order=order,cutoff=cutoff)
        return Reference(result.mean,result.response,result.log_mass,result.metadata)
    raise ValueError(dataset)


def combine_gaussians(query,metric,variances,weights,components):
    means=np.stack([c.mean for c in components])
    logits=np.log(weights)+np.array([c.log_mass for c in components])
    log_mass=logsumexp(logits);probability=np.exp(logits-log_mass)
    mean=probability@means
    gradient=(means-query)@metric/variances[:,None]
    trace=np.array([c.response for c in components])
    # Covariance of component means with their log-evidence gradients.
    response=float(probability@(trace+np.sum((means-mean)*gradient,axis=1)))
    return Reference(mean,response,float(log_mass),{})


def reference(dataset,query,raw_rms_scale,ambient,df,*,metric=None,
              gamma_order=16,spatial_order=64,cutoff=10.):
    query=np.asarray(query,dtype=float)
    metric=np.eye(len(query)) if metric is None else np.asarray(metric,dtype=float)
    if df is None:
        return gaussian_reference(dataset,query,raw_rms_scale,metric=metric,
            order=spatial_order,cutoff=cutoff)
    if df<=2 or raw_rms_scale<=0:raise ValueError('finite-variance Student kernel required')
    nodes,weights=gamma_rule((ambient+df)/2,gamma_order)
    keep=weights>0;nodes=nodes[keep];weights=weights[keep]
    variances=(df-2)*raw_rms_scale**2/(2*nodes)
    components=[gaussian_reference(dataset,query,math.sqrt(v),metric=metric,
        order=spatial_order,cutoff=cutoff) for v in variances]
    result=combine_gaussians(query,metric,variances,weights,components)
    result.metadata=dict(kernel='radial_student_t',ambient=ambient,df=df,
        raw_rms_scale=raw_rms_scale,gamma_order=gamma_order,spatial_order=spatial_order,
        cutoff=cutoff,mixture_weight_derivative_included=True,
        maximum_spatial_nodes=max(c.metadata.get('nodes',c.metadata.get('quadrature_nodes',0)) for c in components))
    return result
