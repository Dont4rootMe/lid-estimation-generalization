"""Run real canonical short cells; these cannot enter benchmark aggregation."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
out=root/'artifacts/non_gaussian_20260911/preflights'
out.mkdir(parents=True,exist_ok=True)
for variant in ('t_flowmatching','pfgmpp'):
    for representation in ('coefficients','dataset'):
        name=f'{variant}__exp__{representation}'
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,'-m','experiments.fair_campaign','run',
                '--variant',variant,'--cell',f'e6/e6_exp_pca/{representation}',
                '--canonical-root',str(root/'artifacts/non_gaussian_20260911/data/benchmarks'),
                '--output',str(out/name),'--steps','20','--preflight'],
                cwd=root,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(cell=name,returncode=p.returncode)),flush=True)
        if p.returncode:raise SystemExit(p.returncode)
