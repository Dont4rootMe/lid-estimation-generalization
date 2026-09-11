"""Package the reviewed patch, fixed weights, pointwise proof and receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from experiments.lambda_repair_proof import PAIRS


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();files={}
    def add(path,name):
        if path.is_file():files[name]=path
    for path in args.report.iterdir():
        if path.suffix in {'.md','.pdf','.png','.csv','.json','.patch'} and 'interim' not in path.name:
            add(path,'report/'+path.name)
    fixed={pair[2] for pair in PAIRS}
    original={pair[1] for pair in PAIRS}
    fixed.update(['nf_exp_lowfreq_terminal32','nf_exp_lowfreq_terminal128','nf_spaghetti_smooth_terminal32'])
    fixed.update(['directrect_exp_spectral_terminal32','directlog_exp_spectral_terminal32','postlog_exp_spectral_terminal32'])
    original.update(['nf_exp_original32','nf_exp_original128','nf_spaghetti_original32'])
    for name in sorted(fixed|original):
        run=args.runs/name
        if not (run/'complete.json').exists():raise RuntimeError('required trained run missing: '+name)
        for filename in ['manifest.json','complete.json','history.jsonl']:
            add(run/filename,'runs/'+name+'/'+filename)
        if name in fixed:add(run/'model.pt','runs/'+name+'/model.pt')
        for path in (run/'source').rglob('*'):add(path,'runs/'+name+'/'+str(path.relative_to(run)))
        for directory in ['evaluation_b512_p16','evaluation_b512_p0','evaluation_nf_float32',
                          'evaluation_nf_float64_exact','denoising','continuous_oracle','nf_heat','jacobian_accounting']:
            for path in (run/directory).rglob('*'):
                add(path,'runs/'+name+'/'+str(path.relative_to(run)))
    for name in ['bridge_spaghetti_original32_run','postrect_spiral_original32']:
        for path in (args.runs/name/'oracle_selection').rglob('*'):
            add(path,'runs/'+name+'/'+str(path.relative_to(args.runs/name)))
    for name in ['legacy_parity_final.json','constant_prefix_parity.json','canonical_training_alignment.json',
                 'canonical_training_alignment_full.json','oracle_selector_sensitivity.json','capacity_control.json',
                 'spiral_covariance_guard_audit.json']:
        add(args.runs/name,'audit/'+name)
    rows=[]
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as archive:
        for name,path in sorted(files.items()):
            value=path.read_bytes();archive.writestr(name,value)
            rows.append(dict(path=name,size=len(value),sha256=hashlib.sha256(value).hexdigest()))
        manifest=dict(schema_version=1,fixed_weights=sorted(fixed),original_controls=sorted(original),
            original_weights_included=False,training_data_included=False,
            contents='fixed native checkpoints; original and repaired pointwise evaluations; reviewed patch and report',
            omitted='large training data, optimizer progress and original weights remain in the source worktree',files=rows)
        archive.writestr('bundle_manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for row in rows:
            assert hashlib.sha256(archive.read(row['path'])).hexdigest()==row['sha256']
    report=dict(status='verified',files=len(rows)+1,fixed_checkpoints=len(fixed),
        zip_bytes=args.output.stat().st_size,zip_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        all_members_checked=True,original_weights_included=False,training_data_included=False)
    (args.report/'bundle_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
