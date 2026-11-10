"""Fresh duplicate-deal tests with a paired common-opponent contrast."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import multiprocessing as mp
import time
import numpy as np
import torch
from fhp_vr_deep.game import load_fhp_game
from fhp_vr_deep.io_utils import sha256_file
from .config import PINS
from .data import verify,local_file,write_json,digest
from .fitting import native_path
from .models import load_policy

_CACHE={}


def policy(path,kind):
    key=(str(path),kind)
    if key not in _CACHE:
        if len(_CACHE)>=8:_CACHE.clear()
        _CACHE[key]=load_policy(path,kind)
    return _CACHE[key]


def hand(game,policies,chance_seed,action_seed):
    chance=np.random.default_rng(chance_seed);actions=np.random.default_rng(action_seed)
    state=game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            a,p=zip(*state.chance_outcomes());selected=int(chance.choice(a,p=p))
        else:
            player=state.current_player();a=state.legal_actions()
            probs=policies[player].action_probabilities(state)
            p=np.asarray([probs.get(x,0.) for x in a],dtype=np.float64)
            if not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:raise ValueError('Invalid action distribution')
            selected=int(actions.choice(a,p=p/p.sum()))
        state.apply_action(selected)
    return state.returns()


def duplicate_values(game,left,right,n,seed):
    # Separate chance/action streams preserve duplicate deals even if policies
    # take different numbers of decisions. Exactly two seat-swapped hands/pair.
    rng=np.random.default_rng(seed);out=np.empty(n)
    for i in range(n):
        c,a=map(int,rng.integers(0,2**63-1,2))
        first=hand(game,(left,right),c,a)[0]
        second=hand(game,(right,left),c,a)[1]
        out[i]=.5*(first+second)*10. # 100 chips/big blind -> mbb/hand
    return out


def moments(x):
    return dict(n=len(x),sum=float(x.sum()),sum_sq=float(np.dot(x,x)))


def planned_tasks(cfg):
    tasks=[]
    for s in cfg['seeds']:
        for rep in cfg['replicates']:
            for kind,u in [('self',None)]+[('anchor',u) for u in cfg['seeds']]:
                for shard,start in enumerate(range(0,cfg['pairs_per_cell'],cfg['shard_pairs'])):
                    seed=int(np.random.SeedSequence([cfg['deal_seed_base'],s,rep,0 if u is None else u+1,shard]).generate_state(1)[0])
                    tasks.append(dict(id=f'{kind}_s{s}_r{rep}_u{u}_k{shard:03d}',kind=kind,seed=s,
                        replicate=rep,anchor_seed=u,shard=shard,deals=min(cfg['shard_pairs'],cfg['pairs_per_cell']-start),deal_seed=seed))
    return tasks


def check_fit(output,seed,cfg):
    worker=Path(output)/'fits'/str(seed)
    receipt=json.loads((worker/'SUCCESS.json').read_text())
    if receipt['contract_sha256']!=digest(cfg):raise ValueError('Different fit contract')
    expected={'result.json'}|{f'{arm}_rep{r}.pt' for arm in cfg['arms'] for r in cfg['replicates']}
    if set(receipt['files'])!=expected:raise ValueError('Incomplete fit receipt')
    for name,sha in receipt['files'].items():
        if sha256_file(worker/name)!=sha:raise ValueError('Fit artifact checksum mismatch')
    result=json.loads((worker/'result.json').read_text())
    if result['seed']!=seed or result['smoke']!=cfg['smoke'] or result['contract_sha256']!=digest(cfg):raise ValueError('Wrong fit identity')
    return receipt


def initialise():
    torch.set_num_threads(1)


def execute(item):
    task,root,output,cfg,fit_hash=item
    path=Path(output)/'matches'/f"{task['id']}.json"
    identity=digest(dict(task=task,contract=digest(cfg),fit_hash=fit_hash))
    if path.exists():
        row=json.loads(path.read_text())
        if row.get('identity')!=identity:raise ValueError('Incompatible match resume')
        saved=row.pop('checksum')
        if digest(row)!=saved:raise ValueError('Corrupt match output')
        row['checksum']=saved;return row
    start=time.perf_counter();game=load_fhp_game();s=task['seed'];r=task['replicate']
    fitted=policy(Path(output)/'fits'/str(s)/f'ucv_refit_rep{r}.pt','refit')
    archived=policy(native_path(root,s,cfg['smoke']),'vr')
    control=policy(Path(output)/'fits'/str(s)/f'native_refit_rep{r}.pt','refit')
    if task['kind']=='self':
        first=duplicate_values(game,fitted,archived,task['deals'],task['deal_seed'])
        second=duplicate_values(game,fitted,control,task['deals'],task['deal_seed'])
        scores={'adapted_vs_archived':moments(first),'adapted_vs_native_refit':moments(second)}
    else:
        u=task['anchor_seed']
        if cfg['smoke']:
            anchor=policy(Path(output)/'fits'/str(s)/'ucv_refit_rep0.pt','refit')
        else:anchor=policy(local_file(root,'ucv',u,PINS['ucv'][str(u)]['policy']),'ucv')
        first=duplicate_values(game,fitted,anchor,task['deals'],task['deal_seed'])
        second=duplicate_values(game,archived,anchor,task['deals'],task['deal_seed'])
        third=duplicate_values(game,control,anchor,task['deals'],task['deal_seed'])
        scores={'adapted_vs_ucv':moments(first),'archived_vs_ucv':moments(second),
                'native_refit_vs_ucv':moments(third),'paired_payoff_gain':moments(first-second),
                'paired_gain_over_native_refit':moments(first-third),
                'native_refit_gain':moments(third-second)}
    row=dict(identity=identity,task=task,scores=scores,elapsed_seconds=time.perf_counter()-start)
    row['checksum']=digest(row);write_json(path,row)
    return row


def evaluate(root,output,cfg,workers=8,profile=False):
    if not cfg['smoke']:
        for s in cfg['seeds']:verify(root,'vr',s);verify(root,'ucv',s)
    fit_hash=digest({s:check_fit(output,s,cfg) for s in cfg['seeds']})
    tasks=planned_tasks(cfg)
    if profile:
        tasks=[dict(t,id='profile_'+t['id'],deals=128) for t in tasks if t['seed']==cfg['seeds'][0] and t['shard']==0][:8]
    items=[(t,str(root),str(output),cfg,fit_hash) for t in tasks]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialise) as pool:
        rows=[]
        for i,row in enumerate(pool.map(execute,items,chunksize=1)):
            rows.append(row)
            if (i+1)%20==0:print(f'Matches {i+1}/{len(tasks)}',flush=True)
    if profile:
        worker_seconds=sum(r['elapsed_seconds'] for r in rows)
        pairs=sum(r['task']['deals']*(2 if r['task']['kind']=='self' else 3) for r in rows)
        estimate=worker_seconds/pairs*sum(t['deals']*(2 if t['kind']=='self' else 3) for t in planned_tasks(cfg))/workers
        result=dict(profile=True,pairs=pairs,worker_seconds=worker_seconds,workers=workers,
                    projected_scoring_hours=estimate/3600,elapsed_seconds=time.perf_counter()-start,
                    caveat='Short workload projection; excludes startup, contention, downloads and fitting.')
        write_json(Path(output)/'analysis/match_timing.json',result);return result
    return rows
