"""Query bootstrap of all-grid selection with an exact population posterior."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.lambda_repair_eval import selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for name in ['bridge_spaghetti_original32_run','postrect_spiral_original32']:
        arrays=np.load(a.runs/name/'oracle_selection/arrays.npz');target=arrays['target']
        for readout in ['full_exact','full_p16','full_p64']:
            curve=arrays[readout];scale,col,diag=selected(curve,target,'full_support_v1')
            rng=np.random.default_rng(921);boot=[]
            for _ in range(500):
                ids=rng.integers(len(target),size=len(target));boot.append(selected(curve[ids],target[ids],'full_support_v1')[0])
            values,counts=np.unique(boot,return_counts=True)
            row=dict(run=name,readout=readout,protocol='full_support_v1',holdout_n=len(target),
                selected_lambda=scale,holdout_mae=float(abs(curve[:,col]-target).mean()),
                bootstrap=[dict(lambda_value=float(v),n=int(n)) for v,n in zip(values,counts)])
            if name.startswith('postrect') and readout=='full_exact':
                delta=abs(curve[:,0]-target)-abs(curve[:,col]-target)
                ids=rng.integers(len(target),size=(2000,len(target)))
                row['minimum_grid_minus_selected_mae']=float(delta.mean())
                row['query_ci95_of_difference']=np.quantile(delta[ids].mean(1),[.025,.975]).tolist()
            rows.append(row)
    a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))


if __name__=='__main__':main()
