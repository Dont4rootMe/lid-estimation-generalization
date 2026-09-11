"""Fixed 12-cell campaign; two independent model workers per A100."""
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
OUT=ROOT/'artifacts/non_gaussian_20260911/full_budget'
HERE=Path(__file__).resolve().parent


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    pending=queue.Queue()
    cells=[]
    for representation in ('coefficients','dataset'):
        for variant in ('t_flowmatching','pfgmpp','posterior_rectified_flow'):
            for task,suite,dataset in [('exp','e6','e6_exp_pca'),('spiral','e1','e1_spiral_pca')]:
                row=dict(name=f'{variant}__{task}__{representation}',variant=variant,
                    cell=f'{suite}/{dataset}/{representation}')
                cells.append(row);pending.put(row)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('freeze a clean producing commit before the campaign')
    (OUT/'campaign.json').write_text(json.dumps(dict(created_unix=time.time(),
        producing_commit=commit,protocol_sha256=hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest(),
        cells=cells,workers_per_gpu=2,gpus=[0,1],independent_retraining_seed_count=1),indent=2)+'\n')
    lock=threading.Lock();failures=[]
    def event(row):
        with lock:
            with (OUT/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(unix=time.time(),**row))+'\n')
            print(json.dumps(row),flush=True)
    def worker(gpu,slot):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        while True:
            try:row=pending.get_nowait()
            except queue.Empty:return
            event(dict(event='start',gpu=gpu,slot=slot,**row))
            with (OUT/(row['name']+'.log')).open('w') as stream:
                process=subprocess.Popen([sys.executable,'-m','experiments.fair_campaign','run',
                    '--variant',row['variant'],'--cell',row['cell'],
                    '--canonical-root',str(ROOT/'artifacts/non_gaussian_20260911/data/benchmarks'),
                    '--output',str(OUT/row['name'])],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                event(dict(event='pid',gpu=gpu,slot=slot,pid=process.pid,**row))
                code=process.wait()
            event(dict(event='exit',gpu=gpu,slot=slot,returncode=code,**row))
            if code:
                with lock:failures.append(dict(**row,returncode=code))
            pending.task_done()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(worker,gpu,slot) for gpu in (0,1) for slot in (0,1)]
        for future in futures:future.result()
    (OUT/'launcher_complete.json').write_text(json.dumps(dict(finished_unix=time.time(),
        status='failed' if failures else 'complete',failures=failures),indent=2)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
