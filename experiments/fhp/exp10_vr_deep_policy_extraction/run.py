"""CLI for pinned replay acquisition, extraction fits, gameplay and analysis."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import platform
import time
import numpy as np
import torch
from fhp_vr_deep.io_utils import sha256_file
from .config import contract
from .data import fetch,write_json,digest
from .fitting import fit_worker,benchmark
from .evaluation import evaluate,check_fit
from .analysis import aggregate


def ensure_run(output,cfg):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[3]
    files={str(p.relative_to(root)):sha256_file(p) for folder in [Path(__file__).parent,root/'fhp_vr_deep',root/'vr_deep_cfr'] for p in folder.rglob('*.py')}
    identity=dict(contract=cfg,implementation=digest(files),runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__))
    path=output/'experiment.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Output belongs to a different implementation/config/runtime; use a new directory')
    write_json(path,identity)


def one_fit(item):
    root,output,seed,cfg=item;torch.set_num_threads(1)
    if (Path(output)/'fits'/str(seed)/'SUCCESS.json').exists():
        check_fit(output,seed,cfg);return {'seed':seed,'resumed_complete':True}
    return fit_worker(root,output,seed,cfg)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['fetch','fit','evaluate','profile-matches','aggregate','all','smoke','benchmark','contract'])
    p.add_argument('--input',type=Path,default=Path('cloud_outputs/vr10-inputs'))
    p.add_argument('--output',type=Path,default=Path('cloud_outputs/vr10-diagnosis'))
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,choices=[0,1,2])
    p.add_argument('--benchmark-steps',type=int,default=100)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv)
    if a.workers<1 or a.benchmark_steps<1:p.error('Positive workers and benchmark steps required')
    smoke=a.smoke or a.mode=='smoke';cfg=contract(smoke)
    if a.mode=='contract':print(json.dumps(cfg,indent=2));return
    if a.mode=='fetch':
        if smoke:p.error('No production downloads in smoke mode')
        fetch(a.input,vr_seeds=cfg['seeds'] if a.seed is None else [a.seed]);return
    torch.set_num_threads(1)
    if a.mode=='benchmark':
        result=benchmark(cfg,a.benchmark_steps);write_json(a.output/'benchmark.json',result);print(json.dumps(result,indent=2));return
    ensure_run(a.output,cfg)
    if a.mode in ['all','smoke','fit']:
        seeds=cfg['seeds'] if a.seed is None else [a.seed]
        if smoke and seeds!=[0]:p.error('Smoke seed is zero')
        if smoke:a.input=a.output/'fixtures'
        with ProcessPoolExecutor(max_workers=min(3,a.workers,len(seeds)),mp_context=mp.get_context('spawn')) as pool:
            for result in pool.map(one_fit,[(str(a.input),str(a.output),s,cfg) for s in seeds]):
                print(f"Fit seed {result['seed']} complete",flush=True)
    if a.mode in ['all','smoke','evaluate','profile-matches']:
        if smoke:a.input=a.output/'fixtures'
        result=evaluate(a.input,a.output,cfg,workers=a.workers,profile=a.mode=='profile-matches')
        if a.mode=='profile-matches':print(json.dumps(result,indent=2))
    if a.mode in ['all','smoke','aggregate']:
        print(json.dumps(aggregate(a.output,cfg),indent=2))


if __name__=='__main__':main()
