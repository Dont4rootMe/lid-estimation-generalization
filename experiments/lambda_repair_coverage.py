"""List actual fixed checkpoints for all eleven native interfaces."""
import argparse
import csv
import json
from pathlib import Path


REPRESENTATIVES={
 'vp_diffusion':('vp_exp_antithetic32','vp_exp_original32'),
 've_diffusion':('ve_exp_antithetic_terminal128','ve_exp_original128'),
 'rectified_flow':('rf_spaghetti_antithetic_terminal128','rf_spaghetti_original128'),
 'scale_conditioned_nf':('nf_exp_lowfreq_terminal128','nf_exp_original128'),
 'schrodinger_bridge':('bridge_spaghetti_antithetic_terminal128','bridge_spaghetti_original128'),
 'direct_rectified_flow':('directrect_exp_spectral_terminal32',None),
 'posterior_rectified_flow':('postrect_spiral_spectral_terminal32_run','postrect_spiral_original32'),
 'direct_log_noise_affine_flow':('directlog_exp_spectral_terminal32',None),
 'posterior_log_noise_affine_flow':('postlog_exp_spectral_terminal32',None),
 'direct_vp_trigonometric_flow':('directvp_exp_spectral_terminal32','directvp_exp_original32'),
 'posterior_vp_trigonometric_flow':('postvp_exp_spectral_terminal32','postvp_exp_original32'),
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for variant,(name,parent) in REPRESENTATIVES.items():
        run=a.runs/name;manifest=json.loads((run/'manifest.json').read_text());done=json.loads((run/'complete.json').read_text())
        nf=variant=='scale_conditioned_nf'
        directory='evaluation_nf_float64_exact' if nf else 'evaluation_b512_p0'
        result=json.loads((run/directory/'test_metrics.json').read_text())
        assert manifest['variant']==variant and result['checkpoint_sha256']==done['checkpoint_sha256']
        rows.append(dict(variant=variant,run=name,dataset=manifest['dataset'],representation=manifest['representation'],
            steps=done['metrics']['steps_completed'],target_lid=result['target_lid'],lambda_value=result['selected_lambda'],
            test_mae=result['test_mae'],readout='exact log-scale derivative' if nf else 'full with exact learned-span trace',
            comparison='matched-budget original/fixed pair' if parent else 'independent native training plus canonical-field alignment',
            original_pair=parent or '',checkpoint_sha256=done['checkpoint_sha256']))
    with a.output.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(trained_native_interfaces=len(rows),matched_source_pairs=sum(bool(x['original_pair']) for x in rows))))


if __name__=='__main__':main()
