"""Independent derivative, kernel identity and quadrature convergence checks."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

from continuous_reference import (Reference,gamma_rule,combine_gaussians,
    gaussian_exp,reference)
from experiments.lambda_repair_reference import tube_posterior,spiral_curve,tube_point


def main():
    rng=np.random.default_rng(702);rows=[]
    # The integral identity and weight derivative are checked against a direct
    # Student posterior on an unrelated finite distribution, including N=784.
    points=rng.normal(size=(31,3));q=points[0].copy()
    for ambient in (30,784):
        for df in (5,128):
            for scale in (.1,1.,10.):
                distance=points-q;d2=np.sum(distance**2,axis=1)
                a=(df-2)*scale**2;p=(df+ambient)/2
                logits=-p*np.log1p(d2/a);weights=np.exp(logits-logsumexp(logits))
                mean=weights@points
                direct_r=float(weights@np.sum((points-mean)*(df+ambient)*distance/(a+d2[:,None]),axis=1))
                nodes,w=gamma_rule(p,64);variance=a/(2*nodes);components=[]
                for v in variance:
                    logw=-d2/(2*v);mass=logsumexp(logw)-np.log(len(points))
                    pr=np.exp(logw-logsumexp(logw));mu=pr@points
                    cov=((points-mu)*pr[:,None]).T@(points-mu)
                    components.append(Reference(mu,float(np.trace(cov)/v),float(mass),{}))
                mixed=combine_gaussians(q,np.eye(3),variance,w,components)
                gap=max(float(abs(mixed.mean-mean).max()),abs(mixed.response-direct_r))
                assert gap<2e-9,(ambient,df,scale,gap)
                rows.append(dict(check='direct_kernel_identity',ambient=ambient,df=df,scale=scale,max_gap=gap))
    # New general-metric Exp integration against the established analytic-angle
    # continuous Gaussian reference, at a curved head and a narrow tail.
    for u in (.1,3.,7.8):
        q=tube_point(u,.8)
        for scale in (.0002,.01,.2):
            old=tube_posterior(q,scale,order=256,cutoff=12)
            new=gaussian_exp(q,scale,metric=np.eye(3),order=128,cutoff=12)
            gap=max(float(abs(old.mean-new.mean).max()),abs(old.response-new.response))
            assert gap<2e-6,(u,scale,gap)
            rows.append(dict(check='exp_analytic_angular_parity',u=u,scale=scale,max_gap=gap))
    # Continuous posterior derivative is also computed by independently moving
    # the query, rather than reusing the Gamma-mixture trace expression.
    for dataset,q in [('e6_exp_pca',tube_point(3.,.7)),('e1_spiral_pca',spiral_curve(6.))]:
        scale=.002;df=5;ambient=30
        center=reference(dataset,q,scale,ambient,df,gamma_order=32,spatial_order=96,cutoff=12)
        step=scale*1e-4;trace=0.
        for j in range(len(q)):
            delta=np.eye(len(q))[j]*step
            plus=reference(dataset,q+delta,scale,ambient,df,gamma_order=32,spatial_order=96,cutoff=12)
            minus=reference(dataset,q-delta,scale,ambient,df,gamma_order=32,spatial_order=96,cutoff=12)
            trace+=(plus.mean[j]-minus.mean[j])/(2*step)
        assert abs(trace-center.response)<2e-5,(dataset,trace,center.response)
        rows.append(dict(check='continuous_finite_difference',dataset=dataset,max_gap=abs(trace-center.response)))
    path=Path(__file__).resolve().parents[2]/'artifacts/non_gaussian_20260911/reference_checks.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(status='passed',rows=rows),indent=2)+'\n')
    print(json.dumps(dict(status='passed',checks=len(rows),maximum_gap=max(r['max_gap'] for r in rows))),flush=True)


if __name__=='__main__':main()
