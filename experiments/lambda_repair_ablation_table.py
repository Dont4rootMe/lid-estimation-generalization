"""Export every completed evaluated arm, including unsuccessful candidates."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[];sensitivity=[]
    for run in sorted(a.runs.iterdir()):
        if not (run/'complete.json').exists():continue
        files=list(run.glob('evaluation*/test_metrics.json'))
        if not files:continue
        manifest=json.loads((run/'manifest.json').read_text());complete=json.loads((run/'complete.json').read_text())
        payload=torch.load(run/'model.pt',weights_only=True,map_location='cpu');cfg=payload['training_config']
        for path in files:
            if 'stored_label_debug_error' in path.parent.name:continue
            receipt=json.loads(path.read_text())
            if receipt.get('batch_size')!=512:continue
            arrays=np.load(path.parent/'test_selected.npz');error=abs(arrays['full'][:,0]-arrays['target'])
            np.testing.assert_allclose(error.mean(),receipt['test_mae'],rtol=0,atol=1e-10)
            readout=(receipt['readout'] if 'readout' in receipt else 'exact learned-span trace' if receipt['probes']==0 else 'source P16')
            row=dict(run=run.name,variant=manifest['variant'],dataset=manifest['dataset'],representation=manifest['representation'],
                arm=manifest['arm'],field_backbone=cfg.get('field_backbone','bottleneck_v1'),
                field_residual_scaling=cfg.get('field_residual_scaling','noise_v1'),field_residual_width=cfg.get('field_residual_width',512),
                training_target=cfg.get('training_target','sample_v1'),training_target_start_step=cfg.get('training_target_start_step',0),
                noise_pairing=cfg.get('noise_pairing','iid'),terminal_decay_steps=cfg.get('terminal_decay_steps',0),
                ema_decay=cfg.get('ema_decay'),max_condition_frequency=cfg['max_condition_frequency'],
                covariance_rank=cfg.get('field_projection_rank'),
                steps_completed=complete['metrics']['steps_completed'],native_best_step=complete['metrics']['best_step'],
                native_best_validation_loss=complete['metrics']['best_validation_loss'],
                recorded_segment_wall_seconds=complete['duration_seconds'],
                selector=receipt.get('selector','bounded_v2'),readout=readout,target_lid=float(arrays['target'][0]),
                holdout_n=1000,test_n=len(error),selected_lambda=receipt['selected_lambda'],selection_status=receipt['status'],
                holdout_mae=receipt['holdout_mae'],test_mae=float(error.mean()),
                bootstrap_probability_ge16=receipt['bootstrap_probability_ge16'],checkpoint_sha256=receipt['checkpoint_sha256'])
            (sensitivity if row['selector']=='full_support_v1' else rows).append(row)
    a.output.mkdir(parents=True,exist_ok=True)
    for name,values in [('all_evaluated_arms.csv',rows),('full_support_sensitivity.csv',sensitivity)]:
        if not values:continue
        with (a.output/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    print(json.dumps(dict(ordinary_rows=len(rows),global_selector_diagnostics=len(sensitivity))))


if __name__=='__main__':main()
