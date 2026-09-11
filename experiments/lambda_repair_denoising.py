"""Noise-normalized denoising risk, independent of LID derivative readout.

Every scale uses128 fixed holdout clean observations and eight fresh Gaussian
corruptions per observation. All checkpoints on one data set share the noise.
The sum-coordinate risk is E||q(X+lambda Z)-X||²/lambda²; lower is better.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from models.training import load_checkpoint
from experiments.lambda_repair_eval import CanonicalPosterior
from experiments.lambda_repair_oracle import SCALES


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((a.run/'manifest.json').read_text());result=load_checkpoint(a.run/'model.pt',device='cuda')
    result.model.eval();folder=a.data/manifest['dataset'];ids=np.load(folder/'holdout_indices.npy')[:128]
    raw=np.load(folder/f"train_{manifest['representation']}.npy",mmap_mode='r')[ids]
    clean=((torch.tensor(raw.copy())-result.normalization_mean)/result.normalization_scale).cuda()
    clean=clean.repeat(8,1);generator=torch.Generator(device='cuda').manual_seed(488)
    noise=torch.randn(clean.shape,device='cuda',generator=generator)
    rows=[];risks=[]
    for scale in SCALES:
        if result.family=='scale_conditioned_normalizing_flow':
            with torch.enable_grad():
                y=(clean+float(scale)*noise).detach().requires_grad_(True)
                density=result.model.log_prob(y,float(scale))
                score=torch.autograd.grad(density.sum(),y)[0]
                prediction=y+float(scale)**2*score
        else:
            with torch.no_grad():prediction=CanonicalPosterior(result,float(scale))(clean+float(scale)*noise)
        risk=((prediction.detach()-clean)/float(scale)).square().sum(1).cpu().numpy().reshape(8,128)
        risks.append(risk)
        rows.append(dict(lambda_value=float(scale),mean_risk=float(risk.mean()),
            query_standard_error=float(risk.mean(0).std(ddof=1)/np.sqrt(128)),query_count=128,corruptions_per_query=8))
    dest=a.run/'denoising';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'arrays.npz',query_ids=ids,scales=SCALES,risk=np.stack(risks))
    (dest/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=result.checkpoint_sha256,
        metric='mean sum-coordinate squared posterior denoising error divided by lambda squared; lower is better',
        noise_seed=488,partition='first128 fixed source-train holdout indices',rows=rows),indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
