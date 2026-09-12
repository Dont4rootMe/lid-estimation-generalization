"""No training: test whether q, div(q), d_log_sigma(q) determine clipped Full.

Construct two positive Gaussian-mixture data densities with identical values
of these local observables and different exact density-scale responses.
The one-dimensional fixture is a mathematical counterexample, not a benchmark.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.pfgm_kernel import ClippedPFGMKernel

HERE=Path(__file__).resolve().parent


def main():
    kernel=ClippedPFGMKernel(1)
    means=np.linspace(-2.5,2.5,12);variance=.15
    query=.6;sigma=.8
    def integrands(radius):
        x=query+np.array([-radius,radius])[:,None]
        density=np.exp(-(x-means)**2/(2*variance))/np.sqrt(2*np.pi*variance)
        spatial=-(x-means)/variance*density
        moment=x*density
        moment_spatial=density+x*spatial
        scale=(x-query)*spatial
        moment_scale=(x-query)*moment_spatial
        return np.stack([density,moment,spatial,moment_spatial,scale,moment_scale]).mean(1)
    p,a,py,ay,ps,as_=kernel.expect_radial(integrands,sigma=sigma)
    base=np.ones(len(means))/len(means)
    q=(a@base)/(p@base)
    constraints=np.stack([np.ones_like(p),p,a,ay-q*py,as_-q*ps])
    # The row space encodes equal normalization, p, q, q_y and q_log_sigma.
    _,singular,vt=np.linalg.svd(constraints,full_matrices=True)
    rank=int((singular>singular[0]*1e-11).sum())
    null=vt[rank:]
    direction=null.T@(null@ps)
    direction/=np.max(np.abs(direction))
    step=.8*np.min(base[np.abs(direction)>1e-15]/np.abs(direction[np.abs(direction)>1e-15]))
    weights=[base-step*direction,base+step*direction]
    rows=[]
    for w in weights:
        density=p@w;posterior=(a@w)/density
        response=(ay@w-posterior*(py@w))/density
        qscale=(as_@w-posterior*(ps@w))/density
        full=1+(ps@w)/density
        score=(py@w)/density
        assert abs(full-(response+(posterior-query)*score))<1e-10
        rows.append(dict(weights=w.tolist(),density=density,posterior=posterior,
                         response=response,posterior_log_sigma_derivative=qscale,
                         marginal_score=score,full_density_dilation=full))
    gaps={key:abs(rows[0][key]-rows[1][key]) for key in rows[0] if key!='weights'}
    assert min(min(r['weights']) for r in rows)>0
    assert max(gaps[key] for key in ('density','posterior','response','posterior_log_sigma_derivative'))<1e-10
    assert gaps['full_density_dilation']>1e-4
    # Independent centered differences of the actual mixture log densities.
    def density_at(scale):
        return kernel.expect_radial(lambda radius:np.exp(-(
            query+np.array([-radius,radius])[:,None]-means)**2/(2*variance)).mean(0)
            /np.sqrt(2*np.pi*variance),sigma=scale)
    h=1e-5
    plus=density_at(sigma*np.exp(h));minus=density_at(sigma*np.exp(-h))
    finite_gaps=[]
    for row,w in zip(rows,weights):
        finite=1+(np.log(plus@w)-np.log(minus@w))/(2*h)
        finite_gaps.append(abs(finite-row['full_density_dilation']))
    assert max(finite_gaps)<1e-8
    result=dict(status='passed',training=False,scope='local Student-style closure counterexample, not all possible global reconstruction methods',
                query=query,native_sigma=sigma,data_component_means=means.tolist(),data_component_variance=variance,
                kernel=kernel.record(),cases=rows,gaps=gaps,constraint_rank=rank,
                centered_log_density_derivative_gaps=finite_gaps)
    (HERE/'full_closure_counterexample.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',gaps=gaps,full=[r['full_density_dilation'] for r in rows])))


if __name__=='__main__':main()
