"""Compare independently trained canonical fields, not just selected MAEs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from models.training import load_checkpoint


NAMES=['ve_exp_spectral_terminal32','directvp_exp_spectral_terminal32','postvp_exp_spectral_terminal32',
       'directrect_exp_spectral_terminal32','directlog_exp_spectral_terminal32','postlog_exp_spectral_terminal32']


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    reference=load_checkpoint(a.runs/NAMES[0]/'model.pt',device='cuda');reference.model.double().eval()
    folder=a.runs/'data/e6_exp_pca';ids=np.load(folder/'holdout_indices.npy')
    raw=np.load(folder/'train_dataset.npy',mmap_mode='r')[ids]
    x=((torch.tensor(raw.copy())-reference.normalization_mean)/reference.normalization_scale).double().cuda()
    z=torch.randn(x.shape,generator=torch.Generator(device='cuda').manual_seed(91203),device='cuda',dtype=torch.float64)
    curves=np.load(a.runs/NAMES[0]/'evaluation_b512_p0/holdout.npz');scales=curves['scales'];rows=[]
    for name in NAMES[1:]:
        other=load_checkpoint(a.runs/name/'model.pt',device='cuda');other.model.double().eval()
        values=np.load(a.runs/name/'evaluation_b512_p0/holdout.npz')
        deltas=[(value-dict(other.model.named_parameters())[key]).flatten() for key,value in reference.model.named_parameters()]
        delta=torch.cat(deltas);clean=[];noisy=[]
        with torch.no_grad():
            for scale in scales:
                for inputs,storage in [(x,clean),(x+float(scale)*z,noisy)]:
                    q=reference.model.canonical_posterior(inputs,float(scale))
                    r=other.model.canonical_posterior(inputs,float(scale))
                    storage.append(torch.linalg.vector_norm(q-r,dim=1).cpu().numpy()/scale)
        rows.append(dict(reference=NAMES[0],other=name,steps=32000,holdout_queries=len(x),scales=len(scales),
            maximum_parameter_difference=float(delta.abs().max()),rms_parameter_difference=float(delta.square().mean().sqrt()),
            maximum_full_difference=float(abs(curves['full']-values['full']).max()),
            mean_full_difference=float(abs(curves['full']-values['full']).mean()),
            maximum_clean_posterior_difference_over_lambda=float(np.max(clean)),
            mean_clean_posterior_difference_over_lambda=float(np.mean(clean)),
            maximum_noisy_posterior_difference_over_lambda=float(np.max(noisy)),
            mean_noisy_posterior_difference_over_lambda=float(np.mean(noisy)),
            reference_checkpoint=reference.checkpoint_sha256,other_checkpoint=other.checkpoint_sha256,
            diagnosis='independent native training, same shared canonical recipe; one seed, not parameter equality'))
    a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))


if __name__=='__main__':main()
