"""Read-only prefix diagnosis; never changes training or its selections."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.fair_protocol import build_model, source_identity
from models import training
from models.neural_fields import exact_divergence

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_gaussian_ambient_20260911'


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=BASE/f'prefix_{int(time.time())}';out.mkdir()
    source=source_identity();rows=[]
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    for run in sorted((BASE/'full_budget').glob('*/progress.pt')):
        manifest=json.loads((run.parent/'manifest.json').read_text())
        assert manifest['source_sha256']==source
        payload=torch.load(run,map_location='cpu',weights_only=False)
        cfg=training.TrainingConfig.from_mapping(payload['training_config'])
        model=build_model(manifest['variant'],cfg,manifest['resolved']['geometry']['ambient_dim'])
        model.load_state_dict(payload['best_state'],strict=True);model=model.cuda().eval()
        assert cfg.field_projection_rank is None and not hasattr(model,'basis')
        population=BASE/'population'/f"{manifest['cell']['dataset']}__{manifest['cell']['representation']}"
        with np.load(population/'queries.npz') as z:raw=z['raw'][:2];ids=z['query_ids'][:2]
        normalized=(torch.tensor(raw,dtype=torch.float32)-payload['normalization']['mean'])/payload['normalization']['scale']
        normalized=normalized.cuda()
        with np.load(population/(manifest['variant']+'.npz')) as z:
            scales=z['scales'];oracle=z['response'][:,:2]
        selected=[]
        for scale in [1/256,1/32,.25,1.,8.,32.,64.]:
            index=int(np.flatnonzero(scales==scale)[0])
            response=exact_divergence(model,normalized,scale).detach().cpu().numpy()
            selected.append(dict(lambda_value=scale,response=response.tolist(),oracle=oracle[index].tolist()))
        summary=dict(run=run.parent.name,completed_steps=payload['global_step'],best_step=payload['best_step'],
            query_ids=ids.tolist(),prefix_only=True,scales=selected)
        rows.append(summary)
        torch.save(dict(model_state=payload['best_state'],training_config=payload['training_config'],
            architecture=payload['architecture'],family=payload['family'],normalization=payload['normalization'],
            prefix_metadata=summary),out/(run.parent.name+'.pt'))
        (out/'summary.json').write_text(json.dumps(dict(status='in_progress',rows=rows),indent=2)+'\n')
        print(json.dumps(summary),flush=True)
        del model,payload
    assert source_identity()==source
    (out/'summary.json').write_text(json.dumps(dict(status='complete',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
