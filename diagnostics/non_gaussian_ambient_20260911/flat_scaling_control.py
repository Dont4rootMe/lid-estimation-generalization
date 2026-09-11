"""Paired, independent flat-data diagnostic; never a benchmark result.

Compare the current scalar residual coefficient with an O(1) low-noise
coefficient. The known normal subspace is used only by the synthetic data
generator and evaluation, never passed to either unrestricted network.
Both copies see identical native t-flow draws and optimizer settings.
"""
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from experiments.fair_protocol import build_model, geometry, resolve, source_identity
from models.neural_fields import exact_divergence
from models.non_gaussian_fields import loss, student_unit_rms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/non_gaussian_ambient_20260911/flat_scaling_control'


class Fixture(nn.Module):
    def __init__(self, core, coefficient):
        super().__init__()
        self.core = core
        self.coefficient = coefficient
        self._lid_noise_pairing = 'antithetic_v1'

    def forward(self, y, scale):
        scale = torch.as_tensor(scale, dtype=y.dtype, device=y.device)
        if scale.ndim == 0:
            scale = scale.expand(len(y))
        inverse = torch.rsqrt(1 + scale.square())[:, None]
        residual = self.core(y * inverse, scale.log(), (scale[:, None] * inverse).expand_as(y))
        factor = scale[:, None] * inverse if self.coefficient == 'noise_v1' else inverse
        return y * inverse.square() + factor * residual


def evaluate(model, basis, normal, step):
    model.eval()
    rng = torch.Generator(device='cuda').manual_seed(99813)
    clean = torch.randn((256, 2), generator=rng, device='cuda') @ basis.T * (15 ** .5)
    rows = []
    for scale in [1/256, 1/32, .25, 1., 8.]:
        noise = student_unit_rms(clean.shape, 5, device=clean.device, dtype=clean.dtype, generator=rng)
        with torch.no_grad():
            y = clean + scale * noise
            prediction = model(y, scale)
            initial = y / (1 + scale**2)
            risk = (prediction - clean).square().sum(1) / scale**2
            initial_risk = (initial - clean).square().sum(1) / scale**2
            normal_risk = (prediction @ normal).square().sum(1) / scale**2
        q = clean[:16].clone().requires_grad_()
        output = model(q, scale)
        jac = torch.stack([torch.autograd.grad(output[:, j].sum(), q, retain_graph=True)[0]
            for j in range(30)], dim=1)
        trace = jac.diagonal(dim1=1, dim2=2).sum(1)
        # A clean law supported on a plane has zero posterior output in every
        # normal direction, hence these output Jacobian rows are exactly zero.
        normal_jac = torch.einsum('oi,bik->bok', normal.T, jac)
        rows.append(dict(step=step, coefficient=model.coefficient, lambda_value=scale,
            mean_full_trace=float(trace.mean()), normal_output_jacobian_frobenius_rms=float(normal_jac.square().sum((1,2)).mean().sqrt()),
            full_denoising_risk=float(risk.mean()), initial_denoising_risk=float(initial_risk.mean()),
            normal_output_risk=float(normal_risk.mean()), queries=16, noisy_observations=256))
    model.train()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cfg, _ = resolve('t_flowmatching', geometry('flat_fixture', 'coefficients', [30]),
        device='cuda', steps=4000, preflight=True)
    task_rng = torch.Generator(device='cuda').manual_seed(28517)
    rotation, _ = torch.linalg.qr(torch.randn((30, 30), generator=task_rng, device='cuda'))
    basis, normal = rotation[:, :2], rotation[:, 2:]
    models = []
    for coefficient in ('noise_v1', 'inverse_total_v1'):
        torch.manual_seed(3981)
        core = build_model('t_flowmatching', cfg, 30).core.cuda()
        model = Fixture(core, coefficient).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
        objective_rng = torch.Generator(device='cuda').manual_seed(15813)
        models.append((model, optimizer, objective_rng))
    for (name, a), (other, b) in zip(models[0][0].state_dict().items(), models[1][0].state_dict().items()):
        assert name == other and torch.equal(a, b)
    source = Path(__file__).read_bytes()
    (OUT / 'source.py').write_bytes(source)
    manifest = dict(status='in_progress', steps=4000, seed=3981, data_seed=28517,
        ambient=30, clean_dimension=2, batch_size=256, native_objective='t_flow_noise_mse',
        kernel_df=5, coefficient_modes=['noise_v1','inverse_total_v1'],
        actual_parameters=sum(p.numel() for p in models[0][0].parameters()),
        protocol='paired synthetic diagnostic; no Exp/Spiral data, labels or parameter search',
        exact_normal_posterior=0, analytic_projection_in_network=False,
        common_source_sha256=source_identity(), diagnostic_sha256=hashlib.sha256(source).hexdigest())
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    started = time.time(); rows = []
    for step in range(1, 4001):
        clean = torch.randn((256, 2), generator=task_rng, device='cuda') @ basis.T * (15 ** .5)
        losses = []
        for model, optimizer, objective_rng in models:
            optimizer.zero_grad(set_to_none=True)
            value = loss('student_t_flow', model, clean, cfg, objective_rng)
            assert torch.isfinite(value)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(value.detach()))
        if step % 500 == 0:
            print(json.dumps(dict(step=step, losses=losses, seconds=time.time()-started)), flush=True)
        if step in (1000, 4000):
            for model, _, _ in models:
                rows.extend(evaluate(model, basis, normal, step))
    for model, _, _ in models:
        torch.save(dict(state_dict=model.cpu().state_dict(), coefficient=model.coefficient,
            steps=4000, manifest=manifest), OUT/(model.coefficient+'.pt'))
    assert source_identity() == manifest['common_source_sha256']
    (OUT / 'result.json').write_text(json.dumps(dict(status='complete', seconds=time.time()-started,
        rows=rows, interpretation='normal output and its Jacobian have an exact zero target; trace is diagnostic, not a benchmark-selected MAE'), indent=2)+'\n')
    print(json.dumps(dict(status='complete', rows=[r for r in rows if r['step']==4000])), flush=True)


if __name__ == '__main__':
    main()
