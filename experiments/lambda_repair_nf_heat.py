"""Fixed-checkpoint Gaussian-convolution consistency of a conditional NF.

For a true Gaussian-smoothed law, d_log(lambda) log p equals
lambda^2 * (Laplacian log p + squared score norm). A conditional density
network need not obey this identity merely because each density is normalized.
This diagnostic does not alter the source OLS5 readout or train any model.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from models.training import load_checkpoint,predict_nf_lid_ols5
from datasets.registry import load_registry


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    m=json.loads((a.run/'manifest.json').read_text());result=load_checkpoint(a.run/'model.pt',device='cuda')
    assert result.family=='scale_conditioned_normalizing_flow';result.model.double().eval()
    f=a.data/m['dataset'];ids=np.load(f/'holdout_indices.npy')[:64]
    raw=np.load(f"{f}/train_{m['representation']}.npy",mmap_mode='r')[ids]
    x=((torch.tensor(raw.copy())-result.normalization_mean)/result.normalization_scale).double().cuda()
    registry=load_registry(Path(__file__).resolve().parents[1]/'configs/datasets/registry/paper_benchmarks.yaml',validate_official_coverage=False)
    target=float(registry[m['dataset']].expected_lid);rows=[];arrays=[];scales=[1/256,1/64,1/16,.25,1.,4.,16.,64.]
    for scale in scales:
        q=x.detach().requires_grad_(True);ell=q.new_full((len(q),),np.log(scale),requires_grad=True)
        density=result.model.log_prob(q,ell.exp())
        derivative=torch.autograd.grad(density.sum(),ell,retain_graph=True)[0]
        score=torch.autograd.grad(density.sum(),q,create_graph=True)[0]
        trace=torch.zeros(len(q),dtype=q.dtype,device=q.device)
        for j in range(q.shape[1]):
            trace+=torch.autograd.grad(score[:,j].sum(),q,retain_graph=j+1<q.shape[1])[0][:,j].detach()
        response=q.shape[1]+scale**2*trace
        correction=scale**2*score.detach().square().sum(1)
        spatial=(response+correction).cpu().numpy()
        native=(q.shape[1]+derivative.detach()).cpu().numpy()
        ols=predict_nf_lid_ols5(result,raw,scale,family='scale_conditioned_nf',ols_log_step=.05,batch_size=64)
        rows.append(dict(lambda_value=scale,query_count=len(q),target_lid=target,
            exact_scale_lid_mae=float(np.abs(native-target).mean()),
            ols5_lid_mae=float(np.abs(ols-target).mean()),
            spatial_posterior_full_mae=float(np.abs(spatial-target).mean()),
            heat_identity_residual_mae=float(np.abs(native-spatial).mean()),
            ols5_vs_exact_scale_derivative_mae=float(np.abs(ols-native).mean()),
            spatial_response_mean=float(response.mean()),spatial_correction_mean=float(correction.mean())))
        arrays.append(np.stack([native,ols,spatial]));print(json.dumps(rows[-1]),flush=True)
    dest=a.run/'nf_heat';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'arrays.npz',scales=scales,query_ids=ids,
        exact_scale_ols5_spatial_full=np.stack(arrays))
    (dest/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=result.checkpoint_sha256,
        dtype='float64',query_partition='first64 source-train holdout',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
