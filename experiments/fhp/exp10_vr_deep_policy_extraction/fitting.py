"""Full-replay, fixed-budget fits; empirical target is a fidelity reference only."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from fhp_vr_deep.io_utils import sha256_file
from fhp_vr_deep.rich_ae import make_encoder
from .data import group_replay,load_replay,local_file,write_json,digest
from .config import PINS
from .models import new_model,probabilities,load_policy


def diagnostics(model,g):
    sums=np.zeros(3);model.eval()
    with torch.no_grad():
        for start in range(0,len(g.features),4096):
            sl=slice(start,start+4096);x=torch.from_numpy(g.features[sl]);m=torch.from_numpy(g.masks[sl])
            logs=torch.log_softmax(model(x).masked_fill(m!=1,-1e20),-1).numpy().astype(np.float64)
            target=g.targets[sl].astype(np.float64);pred=np.exp(logs)
            kl=np.sum(target*(np.log(np.maximum(target,1e-30))-logs),axis=1)
            tv=np.abs(pred-target).sum(1)/2
            mse=np.mean((pred-target)**2,axis=1)
            sums+=np.array([np.dot(g.masses[sl],v) for v in [kl,tv,mse]])
    return dict(zip(['weighted_kl','weighted_tv','weighted_mean_target_mse'],(sums/g.masses.sum()).tolist()))


def fit_one(arrays,g,iteration,*,arm,seed,replicate,cfg,steps=None):
    model=new_model(arm,cfg['init_seed_base']+100*seed+replicate)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rates'][arm])
    rng=random.Random(cfg['draw_seed_base']+100*seed+replicate)
    steps=cfg['updates'][arm] if steps is None else steps
    if arm=='native_refit':
        x,y,m,t=arrays;n=len(x)
    else:
        x,y,m=g.features,g.targets,g.masks;n=len(x)
    batch=min(cfg['batch_size'],n);sequence=hashlib.sha256();started=time.perf_counter()
    model.train()
    for step in range(steps):
        ids=np.asarray(rng.sample(range(n),batch)) if batch<n else np.arange(n)
        sequence.update(ids.astype('<i8').tobytes())
        inputs=torch.from_numpy(np.array(x[ids],dtype=np.float32,copy=True))
        target=torch.from_numpy(np.array(y[ids],dtype=np.float32,copy=True))
        mask=torch.from_numpy(np.array(m[ids],dtype=np.float32,copy=True))
        logits=model(inputs).masked_fill(mask!=1,-1e20)
        if arm=='native_refit':
            w=torch.from_numpy(np.array(t[ids],dtype=np.float32,copy=True).reshape(-1,1))
            w=(w/iteration*2)**(cfg['gamma']/2)
            loss=torch.nn.functional.mse_loss(torch.softmax(logits,-1)*w,target*w)
        else:
            # Uniform groups with mass correction: exactly the original weighted
            # row CE objective and scale, not a uniform-over-information-sets loss.
            w=torch.from_numpy((g.masses[ids]*len(g.masses)/g.rows).astype(np.float32))
            loss=(-(target*torch.log_softmax(logits,-1)).sum(1)*w).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fit')
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%1000==0:print(f'seed={seed} rep={replicate} {arm} {step+1}/{steps}',flush=True)
    return model,dict(fitting_seconds=time.perf_counter()-started,updates=steps,
                      sampling_sha256=sequence.hexdigest(),final_loss=float(loss.detach()))


def synthetic_arrays():
    """Real FHP inputs, synthetic labels, explicitly smoke-only."""
    from fhp_vr_deep.game import load_fhp_game
    rng=np.random.default_rng(1910);enc=make_encoder();xs=[];ms=[]
    for _ in range(10):
        state=load_fhp_game().new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                acts,probs=zip(*state.chance_outcomes());state.apply_action(int(rng.choice(acts,p=probs)))
            else:
                xs.append(enc.information_state(state).copy());ms.append(state.legal_actions_mask())
                state.apply_action(int(rng.choice(state.legal_actions())))
    x=np.tile(np.asarray(xs,dtype=np.float32),(2,1));m=np.tile(np.asarray(ms,dtype=np.float32),(2,1))
    y=rng.random(m.shape)*m;y=(y/y.sum(1,keepdims=True)).astype(np.float32)
    t=rng.integers(1,11,(len(x),1)).astype(np.float32)
    return [x,y,m,t],10


def native_path(root,seed,smoke=False):
    return Path(root)/'fixture_native.pt' if smoke else local_file(root,'vr',seed,PINS['vr'][str(seed)]['policy'])


def fit_worker(root,output,seed,cfg):
    worker=Path(output)/'fits'/str(seed);worker.mkdir(parents=True,exist_ok=True)
    if seed not in cfg['seeds']:raise ValueError('Unexpected source seed')
    if cfg['smoke']:
        arrays,iteration=synthetic_arrays();path=native_path(root,seed,True);path.parent.mkdir(parents=True,exist_ok=True)
        model=new_model('native_refit',111)
        torch.save(dict(type='vr_deep_cfr_policy_snapshot',feature_encoder=make_encoder().metadata(),
                        policy_network_layers=[64,64,64],policy_state_dict=model.state_dict()),path)
    else:
        arrays,meta=load_replay(root,seed);iteration=meta['iteration'];path=native_path(root,seed)
    started=time.perf_counter();g=group_replay(*arrays,iteration);group_seconds=time.perf_counter()-started
    archived=load_policy(path,'vr');base=diagnostics(archived.model,g)
    results=[]
    for rep in cfg['replicates']:
        for arm in cfg['arms']:
            file=worker/f'{arm}_rep{rep}.pt'
            model,timing=fit_one(arrays,g,iteration,arm=arm,seed=seed,replicate=rep,cfg=cfg)
            payload=dict(type='vr10_extraction_policy',arm=arm,seed=seed,replicate=rep,
                         feature_encoder=make_encoder().metadata(),policy_state_dict=model.state_dict(),
                         contract_sha256=digest(cfg),source_policy_sha256=sha256_file(path))
            torch.save(payload,file)
            reloaded=load_policy(file,'refit')
            with torch.no_grad():
                probe=torch.from_numpy(g.features[:32]);mask=torch.from_numpy(g.masks[:32])
                if not torch.equal(probabilities(model,probe,mask),probabilities(reloaded.model,probe,mask)):
                    raise ValueError('Saved policy differs from fit')
            results.append(dict(seed=seed,replicate=rep,arm=arm,**timing,
                                fidelity=diagnostics(model,g),policy_sha256=sha256_file(file),file=file.name))
    result=dict(status='complete',contract_sha256=digest(cfg),seed=seed,smoke=cfg['smoke'],
                source_policy_sha256=sha256_file(path),rows=g.rows,groups=len(g.masses),
                grouping_seconds=group_seconds,archived_fidelity=base,results=results,
                reference='weighted empirical replay policy; not an equilibrium or deployable full-game oracle')
    write_json(worker/'result.json',result)
    write_json(worker/'SUCCESS.json',dict(contract_sha256=digest(cfg),files={p.name:sha256_file(p) for p in [worker/'result.json',*[worker/r['file'] for r in results]]}))
    return result


def benchmark(cfg,steps=100):
    arrays,iteration=synthetic_arrays()
    # A full production batch and many distinct groups, without a GB download.
    rng=np.random.default_rng(10);n=8192
    arrays=[np.resize(a,(n,*a.shape[1:])).copy() for a in arrays]
    arrays[0][:,230]=rng.uniform(0,1,n) # timing fixture only
    g=group_replay(*arrays,iteration)
    result={}
    for arm in cfg['arms']:
        fit_one(arrays,g,iteration,arm=arm,seed=0,replicate=99,cfg=cfg,steps=10)
        _,row=fit_one(arrays,g,iteration,arm=arm,seed=0,replicate=99,cfg=cfg,steps=steps)
        result[arm]=dict(seconds_per_update=row['fitting_seconds']/steps,
                        projected_full_fit_seconds=row['fitting_seconds']/steps*cfg['updates'][arm])
    return dict(timing_only=True,threads=torch.get_num_threads(),steps=steps,arms=result,
                projected_all_fit_hours=sum(r['projected_full_fit_seconds'] for r in result.values())*len(cfg['seeds'])*len(cfg['replicates'])/3600,
                excludes='real replay grouping, diagnostics, cloud setup/download and all match evaluation')
