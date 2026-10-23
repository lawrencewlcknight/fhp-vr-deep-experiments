"""VR-only paired-trajectory gameplay and LBR; no UCV or SD match tasks."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import time

import numpy as np
from fhp_evaluation.cohort import digest, execute_tasks, portable
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES
from fhp_evaluation.statistics import sample_summary
from .sources import read, records, sha256, write

PROTOCOL = "vr_exp7_duration_matches_v1"
LOADER = "fhp_vr_deep.evaluation_adapter:load_policy_for_evaluation"
BASE_SEED = 723_048


def build_tasks(rows, *, smoke=False):
    index = {(r["seed"], r["hours"]): r for r in rows if r["family"] == "vr_deep"}
    if len(rows) != 6 or set(index) != {(s, h) for s in range(3) for h in (24, 48)}:
        raise ValueError("Exactly the six VR endpoints are required")
    tasks = []
    def add(kind, a, b=None, opponent=None):
        total = 2 if smoke else {"direct": 50_000, "rule": 10_000, "lbr": 1_000}[kind]
        shard_size = total if smoke else (10 if kind == "lbr" else 5_000)
        match = f"{kind}_{a['id']}_{b['id'] if b else opponent}"
        for shard, start in enumerate(range(0, total, shard_size)):
            identifier = f"{match}_{shard:03d}"
            # Common rule streams across each VR trajectory; independent LBR
            # streams. No per-deal paired inference is fabricated for LBR.
            stream = f"rule_s{a['seed']}_{opponent}_{shard}" if kind == "rule" else identifier
            seed = int(digest([BASE_SEED, smoke, stream, "deal"])[:15], 16)
            task = dict(id=identifier, match_id=match, kind=kind, a=a, loader=LOADER,
                        deals=min(shard_size, total-start), evaluation_seed=seed, shard=shard)
            if b is not None:
                task["b"] = b
            if opponent is not None:
                task["opponent"] = opponent
            if kind == "lbr":
                task.update(rollouts=16 if smoke else 4096,
                            lbr_seed=int(digest([BASE_SEED, smoke, stream, "lbr"])[:15], 16))
            tasks.append(task)
    for i in range(3):
        for j in range(3):
            add("direct", index[i, 48], index[j, 24])
    for r in rows:
        for opp in PUBLISHED_AGENT_NAMES:
            add("rule", r, opponent=opp)
        add("lbr", r, opponent="local_best_response")
    return tasks


def pool_summaries(items):
    """Exact pooling of sufficient statistics; shards are not training runs."""
    n = sum(x["n"] for x in items)
    if n < 2 or not items:
        raise ValueError("At least two paired deals required")
    mean = sum(x["n"] * x["mean"] for x in items) / n
    m2 = sum((x["n"]-1)*x["std"]**2 + x["n"]*(x["mean"]-mean)**2 for x in items)
    variance = max(0., m2/(n-1))
    se = math.sqrt(variance/n)
    return dict(n=n, mean=mean, std=math.sqrt(variance), se=se,
                conditional_ci95_low=mean-1.96*se, conditional_ci95_high=mean+1.96*se)


def collapse(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["task"]["match_id"]].append(row)
    rows = []
    for name, group in sorted(groups.items()):
        task = group[0]["task"]
        rows.append(dict(match_id=name, kind=task["kind"], seed=task["a"]["seed"],
            hours=task["a"]["hours"], opponent=task.get("opponent"),
            opponent_seed=task.get("b", {}).get("seed"),
            **pool_summaries([r["summary"] for r in group])))
    return rows


def paired_bootstrap(cells, draws=30_000):
    if {(r["seed"], r["opponent_seed"]) for r in cells} != {(i,j) for i in range(3) for j in range(3)} or len(cells) != 9:
        raise ValueError("Nine unique cross-seed cells required")
    means = np.zeros((3,3)); errors = np.zeros((3,3))
    for row in cells:
        means[row["seed"], row["opponent_seed"]] = row["mean"]
        errors[row["seed"], row["opponent_seed"]] = row["se"]
    weights = np.random.default_rng(BASE_SEED).multinomial(3, [1/3]*3, size=draws)/3
    distribution = np.einsum("bi,ij,bj->b", weights, means, weights)
    diagonal = weights @ np.diag(means)
    def result(value, values, se):
        lo, hi = np.quantile(values, [.025, .975])
        return dict(mean_mbb_per_hand=float(value), trajectory_bootstrap_ci95=[float(lo),float(hi)],
                    conditional_deal_se=float(se), conditional_deal_ci95=[float(value-1.96*se),float(value+1.96*se)])
    return dict(all_cross_seed=result(means.mean(), distribution, np.sqrt(np.sum(errors**2))/9),
                same_trajectory=result(np.trace(means)/3, diagonal, np.sqrt(np.sum(np.diag(errors)**2))/3),
                resampling="Same trajectory weights at both horizons; three trajectories, not nine independent runs. "
                           "Bootstrap and conditional deal intervals are separate, exploratory, not joint coverage.",
                bootstrap_draws=draws, seed_pair_matrix=means.tolist())


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarise(results, *, smoke=False):
    cells = collapse(results)
    common, changes = {}, {}
    for kind in ("rule", "lbr"):
        values = {}
        for h in (24,48):
            values[h] = []
            for seed in range(3):
                rs = [r for r in cells if r["kind"]==kind and r["hours"]==h and r["seed"]==seed]
                if len(rs) != (5 if kind=="rule" else 1):
                    raise ValueError("Incomplete common-opponent panel")
                values[h].append(float(np.mean([r["mean"] for r in rs])))
        common[kind] = {str(h):dict(sample_summary(v), seed_values=v) for h,v in values.items()}
        delta = np.asarray(values[48])-values[24]
        changes[kind] = dict(sample_summary(delta), paired_seed_changes=delta.tolist())
    return clean(dict(protocol=PROTOCOL, status="succeeded", smoke=smoke, strength_evidence=not smoke,
        exact_exploitability=False, sd_evaluated=False, ucv_match_evaluated=False,
        direct=paired_bootstrap([r for r in cells if r["kind"]=="direct"]),
        common_opponents=common, paired_48_minus_24_changes=changes, matches=cells,
        duplicate_pairs=sum(r["task"]["deals"] for r in results),
        evaluation_worker_seconds=sum(r["elapsed_seconds"] for r in results),
        interpretation="Positive direct/rule payoff favours VR. LBR is responder payoff: lower is better. "
            "LBR and strategic gaps are not exact exploitability. Paired seed t intervals for common opponents "
            "are supporting, not multiplicity-adjusted. No automatic winner or convergence claim."))


def report(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    write(output / "summary.json", summary)
    direct = summary["direct"]
    lines = ["# VR-Deep 24-to-48-hour evaluation", "",
        "SMOKE: not performance evidence." if summary["smoke"] else "Three paired Experiment 6 trajectories; no new training.", "",
        "| Comparison | Payoff (mbb/hand) | Trajectory bootstrap 95% interval | Conditional deal 95% interval |",
        "| --- | ---: | --- | --- |"]
    for key,label in (("all_cross_seed","All nine seed pairs"),("same_trajectory","Same-trajectory predecessor")):
        row=direct[key]
        interval=lambda k: " to ".join(f"{v:.2f}" for v in row[k])
        lines.append(f"| {label} | {row['mean_mbb_per_hand']:.2f} | "
                     f"{interval('trajectory_bootstrap_ci95')} | {interval('conditional_deal_ci95')} |")
    lines += ["", direct["resampling"], "", summary["interpretation"], "",
        "## Common-opponent changes", "", "```json",
        __import__('json').dumps(summary["paired_48_minus_24_changes"],indent=2), "```", "",
        "Worker-time sums are not job elapsed time. Deployment models are unchanged."]
    (output / "analysis_summary.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    matrix=np.asarray(direct["seed_pair_matrix"])
    im=axes[0].imshow(matrix,cmap="RdBu",vmin=-max(1,abs(matrix).max()),vmax=max(1,abs(matrix).max()))
    for i in range(3):
        for j in range(3):
            axes[0].text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",
                         color="white" if abs(matrix[i,j])>.6*max(1,abs(matrix).max()) else "black")
    axes[0].set(title="48h vs 24h: mbb/hand",xlabel="24h seed",ylabel="48h seed",xticks=range(3),yticks=range(3))
    fig.colorbar(im,ax=axes[0],shrink=.7)
    for ax,kind in zip(axes[1:],('rule','lbr')):
        for s in range(3):
            ax.plot([24,48],[summary['common_opponents'][kind][str(h)]['seed_values'][s] for h in (24,48)],'o-',label=f'seed {s}')
        ax.set(title='Rule payoff (higher better)' if kind=='rule' else 'LBR payoff (lower better)',xlabel='Training hours',ylabel='mbb/hand',xticks=[24,48])
    axes[2].legend();fig.suptitle('SMOKE ONLY' if summary['smoke'] else 'VR-Deep duration diagnostics')
    fig.savefig(output/'duration_comparison.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle-manifest',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--max-hours',type=float,default=40)
    args=parser.parse_args()
    if not 1<=args.workers<=16 or not 0<args.max_hours<=44:parser.error('Invalid resources')
    rows=records(args.input,'matches');tasks=build_tasks(rows,smoke=args.smoke)
    identity=dict(protocol=PROTOCOL,sources=portable(rows),bundle=read(args.bundle_manifest),workers=args.workers,smoke=args.smoke)
    args.output.mkdir(parents=True,exist_ok=True)
    marker=args.output/'SUCCESS.json'
    if marker.exists():
        saved=read(marker)
        if saved['identity']!=digest(identity) or any(sha256(args.output/k)!=v for k,v in saved['files'].items()):raise ValueError('Completed output changed')
        return
    start=time.monotonic();write(args.output/'STATUS.json',dict(status='running',tasks=len(tasks)))
    try:
        results=execute_tasks(tasks,args.output/'scoring',identity,workers=args.workers,seconds=args.max_hours*3600)
        summary=summarise(results,smoke=args.smoke);summary['invocation_seconds']=time.monotonic()-start
        report(args.output,summary)
        files={p.name:sha256(p) for p in args.output.iterdir() if p.is_file() and p.name not in ('STATUS.json','SUCCESS.json')}
        write(marker,dict(status='succeeded',identity=digest(identity),files=files,smoke=args.smoke))
        write(args.output/'STATUS.json',dict(status='succeeded',tasks=len(tasks)))
    except BaseException as error:
        write(args.output/'STATUS.json',dict(status='incomplete',message=str(error),resume='Reuse identical inputs and source bundle; completed shards retained.'))
        raise


if __name__=='__main__':
    main()
