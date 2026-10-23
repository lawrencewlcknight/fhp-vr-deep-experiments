"""Run by file path with ONLY the shared suite on PYTHONPATH.

Native VR and UCV models are queried in isolated shared-suite processes.
No SD-CFR, training, policy modification or whole-game exploitability.
"""
import argparse
import html
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import ROOT, read, records, sha256, write

PROTOCOL = "vr_exp7_remaining_round_v1"


def export_resources(smoke=False):
    return dict(workers=4, cpu_limit=16, task_seconds=1800 if smoke else 7200,
                session_seconds=3600 if smoke else 20*3600)


def library(smoke=False):
    result = read(ROOT / "position_library.json")
    if smoke:
        # Fixed, previously inspected development positions only.
        result["boards"] = [next(r for r in result["boards"]
            if r["partition"] == "development" and r["texture"] == kind)
            for kind in ("unpaired_1suit", "paired_2suit")]
        result["pilot"] = True
        result["partition_counts"] = {"development": 2}
    return result


def make_spec(rows, vr_repo, ucv_repo, partition):
    from fhp_evaluation.strategic_audit import PROTOCOL as suite_protocol
    from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES
    if partition not in ("development", "assessment"):
        raise ValueError("Unknown partition")
    expected = {(a, s, h) for a, hs in (("vr", (24, 48)), ("ucv", (48,)), ("anchor", (24,)))
                for s in range(3) for h in hs}
    if len(rows) != 12 or {(r["cohort"], r["seed"], r["hours"]) for r in rows} != expected:
        raise ValueError("Require all six VR endpoints, three final UCV models and three anchors")
    policies = {name: dict(family="rule", name=name) for name in PUBLISHED_AGENT_NAMES}
    candidates, anchors = [], []
    for r in rows:
        policies[r["id"]] = dict(family=r["family"], checkpoint=r["path"], sha256=r["sha256"],
            repo_root=str(Path(vr_repo if r["family"] == "vr_deep" else ucv_repo).resolve()),
            cohort=r["cohort"], training_seed=r["seed"], hours=r["hours"],
            require_suit_invariance=True, batch_size=1024)
        (anchors if r["cohort"] == "anchor" else candidates).append(r["id"])
    return dict(protocol=suite_protocol, policies=policies, candidates=candidates,
        opponents=list(PUBLISHED_AGENT_NAMES)+anchors,
        range_tilts=[0, -2, 2] if partition == "development" else [0],
        note="Frozen before evaluation. No reference-deal sampling. Assessment is disjoint from recorded "
             "prior diagnostic board classes, not asserted unseen in training. No SD-CFR.")


def seed_summary(values):
    from fhp_evaluation.statistics import sample_summary
    return dict(sample_summary(values), seed_values=list(values))


def summarise(result):
    """Equal opponents within seed; matched horizon differences, independent families."""
    import numpy as np
    if result.get("status") != "succeeded":
        raise ValueError("Incomplete strategic output")
    spec = result["provenance"]["spec"]
    expected = {(a, s, h) for a, hs in (("vr", (24,48)), ("ucv", (48,)))
                for s in range(3) for h in hs}
    observed = {(spec["policies"][c]["cohort"], spec["policies"][c]["training_seed"],
                 spec["policies"][c]["hours"]) for c in spec["candidates"]}
    if observed != expected or len(spec["candidates"]) != 9 or len(spec["opponents"]) != 8:
        raise ValueError("Incomplete strategic cohort")
    endpoints, contrasts = {}, {}
    for tilt in spec["range_tilts"]:
        values, support = {}, {}
        for candidate in spec["candidates"]:
            p = spec["policies"][candidate]
            per_opponent = {}
            for opp in spec["opponents"]:
                row = result["comparisons"][f"{candidate}__vs__{opp}__tilt_{tilt:g}"]
                observed = sorted((tuple(b["board"]), b["root_count"]) for b in row["boards"] if b["core"])
                if not observed or not sum(n for _,n in observed):
                    raise ValueError("Empty root support")
                if opp in support and observed != support[opp]:
                    raise ValueError("Root support differs across candidates")
                support[opp] = observed
                value = row["mean_root_response_gap_bb"]
                if value is None or not math.isfinite(value):
                    raise ValueError("Nonfinite root response gap")
                per_opponent[opp] = value
            values[p["cohort"], p["hours"], p["training_seed"]] = statistics.mean(per_opponent.values())
        panel = {f"{a}_{h}h": [values[a,h,s] for s in range(3)]
                 for a,h in (("vr",24), ("vr",48), ("ucv",48))}
        endpoints[str(tilt)] = {name: seed_summary(v) for name,v in panel.items()}
        delta = np.asarray(panel["vr_48h"])-panel["vr_24h"]
        rng = np.random.default_rng(723048)
        # UCV seed 0 is not a matched VR seed 0.
        vr = np.asarray(panel["vr_48h"])[rng.integers(0,3,size=(30000,3))].mean(axis=1)
        ucv = np.asarray(panel["ucv_48h"])[rng.integers(0,3,size=(30000,3))].mean(axis=1)
        contrasts[str(tilt)] = dict(vr_48_minus_24=seed_summary(delta),
            vr48_minus_ucv48=dict(mean=float(np.mean(panel["vr_48h"])-np.mean(panel["ucv_48h"])),
                independent_seed_bootstrap_ci95=np.quantile(vr-ucv,[.025,.975]).tolist()))
    return dict(protocol=PROTOCOL, partition=result["provenance"]["partition"],
        endpoints=endpoints, contrasts=contrasts, units="BB per supported audited root",
        primary_tilt=0, lower_is_better=True, exact_exploitability=False,
        interpretation="Conditional remaining-round response gap, not full-game exploitability. Equal weights "
          "across eight fixed opponents within each seed. VR horizon contrasts pair the same three trajectories; "
          "UCV/VR seeds are independent. Intervals are exploratory with only three training trajectories. "
          "Do not pool development and assessment or treat positions as independent training replicates.")


def write_report(output, summaries, smoke):
    write(output / "summary.json", dict(protocol=PROTOCOL, status="succeeded",
        smoke=smoke, strength_evidence=not smoke, partitions=summaries,
        assessment_note=read(ROOT/"position_library.json")["assessment_note"]))
    parts = ["<!doctype html><meta charset='utf-8'><title>VR duration strategic audit</title>",
        "<h1>VR-Deep duration and strategic diagnostics</h1>",
        "<p>SMOKE: correctness only.</p>" if smoke else
        "<p>Frozen 24/48-hour VR policies and final 48-hour UCV policies. No SD-CFR.</p>"]
    for partition, summary in summaries.items():
        parts += [f"<h2>{partition}</h2>", "<p>"+html.escape(summary["interpretation"])+"</p>",
            f'<p><a href="{partition}/analysis/report.html">Detailed strategic positions and graphics</a></p>',
            "<table><tr><th>Policy</th><th>Mean root gap (BB)</th><th>Seed values</th></tr>"]
        for name, row in summary["endpoints"]["0"].items():
            parts.append(f"<tr><td>{name}</td><td>{row['mean']:.6f}</td><td>{row['seed_values']}</td></tr>")
        parts += ["</table>", "<p>Paired VR 48-minus-24 root-gap change: "
            +html.escape(str(summary["contrasts"]["0"]["vr_48_minus_24"]))+"</p>"]
        parts.append("<p>VR 48h minus UCV 48h root-gap difference (independent training seeds): "
            +html.escape(str(summary["contrasts"]["0"]["vr48_minus_ucv48"]))+"</p>")
        # Examples are illustrations, not additional confirmatory tests.
        parts.append("<p>The detailed report includes strongest-hand, pot-odds and final-call examples. "
            "Its largest-error cases are selected after scoring; they are not representative match frequencies "
            "and must not be interpreted as whole-game loss rates.</p>")
    parts.append("<p>"+html.escape(read(ROOT/"position_library.json")["assessment_note"])+"</p>")
    (output/"report.html").write_text("\n".join(parts))


def main(argv=None):
    from fhp_evaluation.cohort import digest
    from fhp_evaluation.strategic_audit.validation import validate
    from fhp_evaluation.strategic_audit.workflow import run
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "vr-repo", "ucv-repo", "suite-repo", "bundle-manifest"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    import fhp_evaluation
    if Path(fhp_evaluation.__file__).resolve().parent.parent != args.suite_repo.resolve():
        raise ValueError("Use the shared suite, not a vendored evaluator")
    rows = records(args.input, "strategic")
    lib = library(args.smoke)
    specs = {p: make_spec(rows,args.vr_repo,args.ucv_repo,p)
             for p in (("development",) if args.smoke else ("development","assessment"))}
    # Fixed resources across restart attempts, because the suite hashes its limits.
    resources = export_resources(args.smoke)
    contract = dict(protocol=PROTOCOL, library=lib, specs=specs, resources=resources,
                    bundle=read(args.bundle_manifest), smoke=args.smoke, threads=4)
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/"contract.json").exists() and read(out/"contract.json") != contract:
        raise ValueError("Changed strategic contract; reuse the original source bundle for resume")
    write(out/"contract.json",contract)
    marker = out/"SUCCESS.json"
    if marker.exists():
        done = read(marker)
        if done["identity"] != digest(contract) or any(sha256(out/k)!=v for k,v in done["files"].items()):
            raise ValueError("Changed completed strategic result")
        return
    write(out/"STATUS.json",dict(status="running"))
    try:
        write(out/"oracle_validation.json",validate())
        summaries = {}
        for partition,spec in specs.items():
            directory = out/partition
            path = run(spec,lib,directory,partition=partition,include_reference=False,
                unlock_assessment=partition=="assessment",resume=(directory/"manifest.json").exists(),
                threads=4,max_seconds=resources["session_seconds"],max_rss_mb=52000,board_exports=resources)
            summaries[partition] = summarise(read(path))
            write(directory/"analysis/duration_summary.json",summaries[partition])
        write_report(out,summaries,args.smoke)
        files = {str(p.relative_to(out)):sha256(p) for p in
                 [out/"summary.json",out/"report.html",out/"oracle_validation.json"]+
                 [out/p/"analysis"/f for p in specs for f in ("summary.json","report.html","duration_summary.json")]}
        write(marker,dict(status="succeeded",identity=digest(contract),files=files,smoke=args.smoke))
        write(out/"STATUS.json",dict(status="succeeded"))
    except BaseException as error:
        write(out/"STATUS.json",dict(status="incomplete",message=str(error),
            resume="Completed tables and analysis boards retained. Resume with the immutable source bundle."))
        raise


if __name__ == "__main__":
    main()
