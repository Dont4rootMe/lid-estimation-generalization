"""Bound oracle changes from the exact float32 query preprocessing path."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
from continuous_reference import reference

ROOT=Path(__file__).resolve().parents[2]/'artifacts/non_gaussian_20260911/population'


def check(job):
    folder,kernel,df=job;folder=Path(folder)
    meta=json.loads((folder/'geometry.json').read_text());rms=meta['normalization_rms']
    with np.load(folder/'queries.npz') as z:
        raw=z['raw'][:4];mean=z['mean'];renderer=z['renderer'];offset=z['offset'];metric=z['metric']
        scales=z['scales'];ids=z['query_ids'][:4];original=z['coefficients'][:4]
    normalized=(torch.tensor(raw,dtype=torch.float32)-torch.tensor(mean))/rms
    observed=normalized.double().numpy()*rms+mean
    active=(observed-offset)@renderer.T@np.linalg.inv(metric)
    normal=observed-(active@renderer+offset);normal_squared=np.sum(normal**2,axis=1)
    with np.load(folder/(kernel+'.npz')) as z:baseline=z['response'][:,:4]
    checked=[]
    for j,scale in enumerate(scales):
        for i,q in enumerate(active):
            effective=np.sqrt((scale*rms)**2+(normal_squared[i]/(df-2) if df else 0.))
            value=reference(meta['dataset'],q,effective,meta['ambient'],df,metric=metric)
            checked.append(dict(lambda_value=float(scale),query_id=int(ids[i]),
                exact_input_response=float(baseline[j,i]),rounded_input_response=value.response,
                absolute_response_gap=abs(float(baseline[j,i])-value.response)))
    return dict(cell=meta['cell_key'],kernel=kernel,checks=checked,
        max_coordinate_shift=float(abs(active-original).max()),
        maximum_response_gap=max(c['absolute_response_gap'] for c in checked))


def main():
    torch.set_num_threads(1)
    jobs=[(str(p),kernel,df) for p in ROOT.iterdir() if p.is_dir()
        for kernel,df in [('t_flowmatching',5),('pfgmpp',128),('gaussian',None)]]
    with ProcessPoolExecutor(max_workers=6) as executor:rows=list(executor.map(check,jobs))
    maximum=max(r['maximum_response_gap'] for r in rows)
    output=dict(status='passed' if maximum<.002 else 'requires_exact_input_references',
        interpretation='effect of the actual float32 input transform on a continuous-law posterior, not model rounding error',
        maximum_response_gap=maximum,rows=rows)
    (ROOT.parent/'query_precision_checks.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(status=output['status'],maximum_response_gap=maximum)),flush=True)


if __name__=='__main__':main()
