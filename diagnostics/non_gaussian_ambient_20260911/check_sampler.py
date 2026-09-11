"""Independent closed-form integration gate, not a learned-quality score."""
import json
from pathlib import Path
import math

import torch

from generation_quality import sample


class GaussianMean(torch.nn.Module):
    def __init__(self,variances):
        super().__init__();self.register_buffer('variances',variances)

    def forward(self,y,scale):return y*self.variances/(self.variances+scale**2)


def main():
    torch.set_num_threads(2)
    variance=torch.tensor([.05,1.,20.],dtype=torch.float64)
    model=GaussianMean(variance)
    initial=torch.tensor([[1.,-2.,3.],[-4.,5.,-6.]],dtype=torch.float64)*64
    exact_endpoint=initial*torch.sqrt((variance+(1/256)**2)/(variance+64**2))
    exact_sample=exact_endpoint*variance/(variance+(1/256)**2)
    rows=[]
    for steps in (512,1024):
        generated,endpoint=sample(model,initial,steps)
        gap=float((generated-exact_sample).abs().max())
        end_gap=float((endpoint-exact_endpoint).abs().max())
        rows.append(dict(steps=steps,maximum_sample_gap=gap,maximum_endpoint_gap=end_gap))
    assert rows[1]['maximum_sample_gap']<.001
    assert rows[1]['maximum_sample_gap']<rows[0]['maximum_sample_gap']/3
    destination=Path(__file__).resolve().parents[2]/'artifacts/non_gaussian_ambient_20260911/sampler_gate.json'
    destination.write_text(json.dumps(dict(status='passed',rows=rows,
        operation='canonical Heun integration and final denoiser versus exact anisotropic Gaussian trajectory'),indent=2)+'\n')
    print(destination.read_text())


if __name__=='__main__':main()
