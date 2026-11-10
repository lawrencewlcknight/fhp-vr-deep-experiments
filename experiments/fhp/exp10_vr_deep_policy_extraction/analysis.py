"""Average fit repetitions before seed inference; reject missing shards."""
from collections import defaultdict
from pathlib import Path
import json
import math
import numpy as np
from scipy.stats import t
from fhp_vr_deep.io_utils import write_csv,sha256_file
from .data import digest,write_json
from .evaluation import planned_tasks,check_fit


def pool_moments(ms):
    n=sum(m['n'] for m in ms);total=sum(m['sum'] for m in ms);ss=sum(m['sum_sq'] for m in ms)
    if n<2 or not all(math.isfinite(v) for v in [total,ss]):raise ValueError('Invalid moments')
    variance=max(0.,(ss-total*total/n)/(n-1))
    return dict(n=n,mean_mbb=total/n,conditional_mc_se_mbb=math.sqrt(variance/n))


def bootstrap(matrix,seed=1010):
    rng=np.random.default_rng(seed);n,m=matrix.shape
    if n<2 or m<2:return dict(mean=float(matrix.mean()),ci_low=None,ci_high=None)
    rows=rng.integers(n,size=(10000,n));cols=rng.integers(m,size=(10000,m))
    draws=matrix[rows[:,:,None],cols[:,None,:]].mean(axis=(1,2))
    return dict(mean=float(matrix.mean()),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)))


def paired_summary(values):
    x=np.asarray(values,dtype=float);n=len(x);mean=float(x.mean())
    if n<2:return dict(mean=mean,ci_low=None,ci_high=None,n=n)
    half=float(t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n))
    return dict(mean=mean,ci_low=mean-half,ci_high=mean+half,n=n)


def aggregate(output,cfg):
    output=Path(output);fits={s:check_fit(output,s,cfg) for s in cfg['seeds']};fit_hash=digest(fits)
    cells=defaultdict(list)
    for task in planned_tasks(cfg):
        row=json.loads((output/'matches'/f"{task['id']}.json").read_text())
        checksum=row.pop('checksum')
        expected=digest(dict(task=task,contract=digest(cfg),fit_hash=fit_hash))
        if digest(row)!=checksum or row['identity']!=expected or row['task']!=task:raise ValueError('Incompatible/corrupt match shard')
        metrics={'adapted_vs_archived','adapted_vs_native_refit'} if task['kind']=='self' else {'adapted_vs_ucv','archived_vs_ucv','native_refit_vs_ucv','paired_payoff_gain','paired_gain_over_native_refit','native_refit_gain'}
        if set(row['scores'])!=metrics:raise ValueError('Incomplete metrics')
        for metric,stats in row['scores'].items():
            if stats['n']!=task['deals']:raise ValueError('Incomplete deal count')
            cells[(metric,task['seed'],task['replicate'],task['anchor_seed'])].append(stats)
    cell_rows=[dict(metric=k[0],seed=k[1],replicate=k[2],anchor_seed=k[3],**pool_moments(v)) for k,v in sorted(cells.items())]
    if any(r['n']!=cfg['pairs_per_cell'] for r in cell_rows):raise ValueError('Incomplete cell')
    summaries={}
    for metric in ['adapted_vs_ucv','archived_vs_ucv','native_refit_vs_ucv','paired_payoff_gain','paired_gain_over_native_refit','native_refit_gain']:
        matrix=np.array([[np.mean([r['mean_mbb'] for r in cell_rows if r['metric']==metric and r['seed']==s and r['anchor_seed']==u]) for u in cfg['seeds']] for s in cfg['seeds']])
        rows=[r for r in cell_rows if r['metric']==metric]
        summaries[metric]=dict(**bootstrap(matrix),conditional_mc_se_mbb=math.sqrt(sum(r['conditional_mc_se_mbb']**2 for r in rows))/len(rows))
    for metric in ['adapted_vs_archived','adapted_vs_native_refit']:
        values=[np.mean([r['mean_mbb'] for r in cell_rows if r['metric']==metric and r['seed']==s]) for s in cfg['seeds']]
        summaries[metric]=paired_summary(values)
    fit_rows=[];fidelity=[]
    for s in cfg['seeds']:
        result=json.loads((output/'fits'/str(s)/'result.json').read_text())
        for r in result['results']:
            fit_rows.append({k:v for k,v in r.items() if k!='fidelity'}|r['fidelity'])
        for metric in ['weighted_kl','weighted_tv','weighted_mean_target_mse']:
            baseline=result['archived_fidelity'][metric]
            means={arm:np.mean([r['fidelity'][metric] for r in result['results'] if r['arm']==arm]) for arm in cfg['arms']}
            fidelity.append(dict(seed=s,metric=metric,archived=baseline,**means,
                                 adapted_minus_archived=means['ucv_refit']-baseline,
                                 adapted_minus_native_refit=means['ucv_refit']-means['native_refit']))
    summary=dict(status='complete',smoke=cfg['smoke'],contract_sha256=digest(cfg),
        primary=summaries['paired_payoff_gain'],comparisons=summaries,
        source_training_seeds=len(cfg['seeds']),fit_repetitions_per_seed=len(cfg['replicates']),
        total_fits=len(fit_rows),duplicate_pairs=sum(t['deals']*(2 if t['kind']=='self' else 3) for t in planned_tasks(cfg)),
        inference='Primary: 95% crossed VR/UCV seed bootstrap after averaging fit repetitions. Exploratory n=3 per cohort. Secondary intervals descriptive.',
        caveats=['This tests a complete extraction package, not grouping alone.',
                 'Replay fidelity is in-sample; full FHP exploitability and exact trajectory average are unavailable.',
                 'Positive primary gain means improved payoff against these fixed UCV opponents, not universal strength.',
                 'Do not estimate a fraction of an historical match gap using different deals/opponents.',
                 'Fit repetitions are not independent self-play training seeds; UCV policies are historical.'])
    out=output/'analysis';out.mkdir(exist_ok=True)
    write_json(out/'summary.json',summary);write_csv(out/'match_cells.csv',cell_rows)
    write_csv(out/'fit_results.csv',fit_rows);write_csv(out/'fidelity_by_seed.csv',fidelity)
    report=['# VR-Deep policy-extraction diagnosis','',summary['inference'],'',
            'Positive payoff differences favour the UCV extraction package. All refits use frozen VR replay.','',
            '| Comparison | Mean mbb/hand | Interval |','|---|---:|---|']
    for name,r in summaries.items():report.append(f"| {name} | {r['mean']:.4f} | [{r['ci_low']}, {r['ci_high']}] |")
    report+=['',*summary['caveats']]
    (out/'report.md').write_text('\n'.join(report)+'\n')
    write_json(out/'SUCCESS.json',dict(contract_sha256=digest(cfg),files={p.name:sha256_file(p) for p in out.iterdir() if p.is_file() and p.name!='SUCCESS.json'}))
    return summary
