"""Independent finite-prior and continuous half-line tests of Student F."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
import torch

from student_full import density_dilation


def check():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(84011)
    rows = []
    for ambient in (2, 6, 30):
        centers = torch.randn((7, ambient), generator=generator, dtype=torch.float64)
        prior = torch.arange(1, 8, dtype=torch.float64)
        query = torch.randn((3, ambient), generator=generator, dtype=torch.float64)
        for df in (5, 128):
            for scale in (.02, .7, 3.):
                def field(q, lam):
                    if lam.ndim == 0:
                        lam = lam.expand(len(q))
                    distance = (q[:, None] - centers).square().sum(2)
                    logit = prior.log() - (ambient + df) / 2 * torch.log1p(
                        distance / ((df - 2) * lam[:, None].square()))
                    return logit.softmax(1) @ centers
                response = []
                for q in query:
                    jac = torch.autograd.functional.jacobian(
                        lambda z: field(z[None], torch.tensor(scale, dtype=query.dtype))[0], q)
                    response.append(jac.trace())
                response = torch.stack(response)
                full, posterior, contraction = density_dilation(field, query, scale, response, df)
                distance = (query[:, None] - centers).square().sum(2)
                a = (df - 2) * scale ** 2
                weight = (prior.log() - (ambient + df) / 2 * torch.log1p(distance / a)).softmax(1)
                # Direct kernel derivative: no posterior derivatives or above identity.
                direct = ((ambient + df) * weight * distance / (a + distance)).sum(1)
                h = 1e-5
                finite = ((query - posterior) * (field(query, torch.tensor(scale * math.exp(h), dtype=query.dtype))
                          - field(query, torch.tensor(scale * math.exp(-h), dtype=query.dtype))) / (2 * h)).sum(1)
                rows.append(dict(ambient=ambient, df=df, scale=scale,
                    density_identity_gap=float((full-direct).abs().max()),
                    scale_derivative_gap=float((contraction-finite).abs().max())))
    # Lebesgue measure on a half-line: density dilation is exactly d=1;
    # response alone is deficient at its endpoint. Numerical integrals are
    # over dimensionless u=x/r, r=lambda*sqrt(nu-2), with no fitted model.
    boundaries = []
    for ambient in (2, 30, 784):
        for df in (5, 128):
            p = (ambient + df) / 2
            mass = quad(lambda u: (1+u*u)**(-p), 0, np.inf, epsabs=1e-12)[0]
            mean = quad(lambda u: u*(1+u*u)**(-p), 0, np.inf, epsabs=1e-12)[0] / mass
            score = quad(lambda u: 2*p*u*(1+u*u)**(-p-1), 0, np.inf, epsabs=1e-12)[0] / mass
            cross = quad(lambda u: 2*p*u*u*(1+u*u)**(-p-1), 0, np.inf, epsabs=1e-12)[0] / mass
            response = cross - mean*score
            # b=r*mean, e=-r*mean, e.dot(d_log(lambda)b)/r**2=-mean**2.
            full = (response + (ambient+df-1)*mean**2)/(1+mean**2)
            boundaries.append(dict(ambient=ambient, df=df, response=response, full=full))
    result = dict(status='passed', finite_prior=rows, homogeneous_boundary=boundaries,
        max_density_identity_gap=max(r['density_identity_gap'] for r in rows),
        max_scale_derivative_gap=max(r['scale_derivative_gap'] for r in rows),
        max_boundary_lid_gap=max(abs(r['full']-1) for r in boundaries))
    assert result['max_density_identity_gap'] < 1e-10, result
    assert result['max_scale_derivative_gap'] < 2e-7, result
    assert result['max_boundary_lid_gap'] < 1e-8, result
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=check();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}))
