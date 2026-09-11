"""Eight requested full-ambient trainings, with separate technical preflights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_gaussian_ambient_20260911'
HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    out=BASE/('preflights' if args.preflight else 'full_budget')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cells=[]
    for representation in ('dataset','coefficients'):
        for variant in ('t_flowmatching','pfgmpp'):
            for task,suite,dataset in [('exp','e6','e6_exp_pca'),('spiral','e1','e1_spiral_pca')]:
                if args.preflight and task=='spiral':continue
                cells.append(dict(name=f'{variant}__{task}__{representation}',variant=variant,
                    cell=f'{suite}/{dataset}/{representation}'))
    if not args.preflight and subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('commit a clean producing source before full training')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (out/'campaign.json').write_text(json.dumps(dict(created_unix=time.time(),producing_commit=commit,
        protocol_sha256=hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest(),
        cells=cells,workers_per_gpu=2,gpus=[0,1],preflight=args.preflight),indent=2)+'\n')
    pending=queue.Queue()
    for row in cells:pending.put(row)
    lock=threading.Lock();failures=[]
    def event(row):
        with lock:
            with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(dict(unix=time.time(),**row))+'\n')
            print(json.dumps(row),flush=True)
    def worker(gpu,slot):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        while True:
            try:row=pending.get_nowait()
            except queue.Empty:return
            command=[sys.executable,'-m','experiments.fair_campaign','run','--variant',row['variant'],
                '--cell',row['cell'],'--canonical-root',str(BASE/'data/benchmarks'),'--output',str(out/row['name'])]
            if args.preflight:command+=['--preflight','--steps','200']
            with (out/(row['name']+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                event(dict(event='start',gpu=gpu,slot=slot,pid=child.pid,**row))
                code=child.wait()
            event(dict(event='exit',returncode=code,**row))
            if code:
                with lock:failures.append(dict(**row,returncode=code))
            pending.task_done()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(worker,gpu,slot) for gpu in (0,1) for slot in (0,1)]
        for future in futures:future.result()
    (out/'launcher_complete.json').write_text(json.dumps(dict(status='failed' if failures else 'complete',
        finished_unix=time.time(),failures=failures),indent=2)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
