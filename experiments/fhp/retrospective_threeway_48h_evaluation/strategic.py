"""Isolated shared-suite strategic audit of the frozen three-way 48h cohort.

Run by file path with ONLY the shared suite on PYTHONPATH. Native policy
families load in separate suite workers; never import the match runner here.
This is a final-round diagnostic, not LBR or whole-game exploitability.
"""

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

ALGORITHMS = ("sd", "ucv", "vr")
LABELS = {"sd": "SD-CFR Exp6", "ucv": "UCV-ESCHER Exp16", "vr": "VR-Deep Exp6"}
FAMILIES = {"sd": "sd_cfr", "ucv": "ucv", "vr": "vr_deep"}
PROTOCOL = "fhp_threeway_48h_strategic_v2_board_exports"


def export_resources(stage, workers=4):
    if stage not in ("pilot", "development") or workers not in (1, 2, 4):
        raise ValueError("Use pilot/development and 1, 2 or 4 export workers")
    return dict(workers=workers, task_seconds=7200,
                session_seconds=16200 if stage == "pilot" else 75600, cpu_limit=16)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_spec(records, repos, anchor_root, stage):
    from fhp_evaluation.strategic_audit import PROTOCOL as SUITE_PROTOCOL
    from fhp_evaluation.strategic_audit.exp16 import checkpoint_registry, RULES
    from fhp_evaluation.loaders import sha256_file
    if stage not in ("pilot", "development"):
        raise ValueError("Only pilot/development are enabled; assessment remains locked")
    if sorted((r["algorithm"], r["seed"]) for r in records) != [
            (a, s) for a in ALGORITHMS for s in range(3)]:
        raise ValueError("Expected the complete nine-policy 48h cohort")
    policies = {name: dict(family="rule", name=name) for name in RULES}
    candidates = []
    for record in records:
        if record["training_hours"] != 48 or sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("Changed or non-48h candidate")
        if stage == "pilot" and record["seed"] != 0:
            continue
        algorithm = record["algorithm"]
        label = f"{algorithm}_seed{record['seed']}_48h"
        candidates.append(label)
        policies[label] = dict(family=FAMILIES[algorithm], checkpoint=str(Path(record["path"]).resolve()),
            sha256=record["sha256"], repo_root=str(Path(repos[algorithm]).resolve()),
            cohort=algorithm, training_seed=record["seed"], hours=48,
            require_suit_invariance=True, batch_size=1024, model_batch_size=32)
    # Reuse the exact Exp16 audit's frozen eight-opponent panel and anchor hashes.
    anchors = []
    for label, record in checkpoint_registry("pilot").items():
        if record["cohort"] != "exp9":
            continue
        anchors.append(label)
        policies[label] = dict(family="ucv", checkpoint=str(Path(anchor_root).resolve() / f"{label}.pkl"),
            sha256=record["sha256"], repo_root=str(Path(repos["ucv"]).resolve()),
            cohort="exp9", training_seed=record["training_seed"], hours=24, require_suit_invariance=True)
    return dict(protocol=SUITE_PROTOCOL, policies=policies, candidates=candidates,
        opponents=list(RULES) + anchors, reference_policy="exp9_seed0_24h", range_tilts=[0, -2, 2],
        note="Only 48h candidates. Independent training seeds across algorithms, not matched pairs. "
             "Same library/opponents as the Exp16 strategic audit; no LBR or assessment access.")


def summarise(result, stage):
    """Equal opponent weights within seed first; independent algorithm cohorts."""
    if result.get("status") != "succeeded":
        raise ValueError("A complete shared-suite result is required")
    spec = result["provenance"]["spec"]
    seeds = (0,) if stage == "pilot" else (0, 1, 2)
    expected = [f"{a}_seed{s}_48h" for a in ALGORITHMS for s in seeds]
    if sorted(spec["candidates"]) != sorted(expected) or spec["range_tilts"] != [0, -2, 2]:
        raise ValueError("Unexpected strategic candidate/tilt panel")
    endpoints = {}
    for tilt in spec["range_tilts"]:
        support = {}
        values = {a: {} for a in ALGORITHMS}
        for algorithm in ALGORITHMS:
            for opponent in spec["opponents"]:
                values[algorithm][opponent] = {}
                for seed in seeds:
                    row = result["comparisons"][f"{algorithm}_seed{seed}_48h__vs__{opponent}__tilt_{tilt:g}"]
                    observed = sorted((tuple(b["board"]), b["root_count"]) for b in row["boards"] if b["core"])
                    if not observed or not sum(n for _, n in observed):
                        raise ValueError("Empty strategic root support")
                    if opponent in support and observed != support[opponent]:
                        raise ValueError("Candidate root support differs; refusing an unfair comparison")
                    support[opponent] = observed
                    value = row["mean_root_response_gap_bb"]
                    if value is None or not math.isfinite(value):
                        raise ValueError("Missing/non-finite root gap")
                    values[algorithm][opponent][str(seed)] = value
            values[algorithm]["equal_weight_panel"] = {
                str(seed): statistics.mean(values[algorithm][opp][str(seed)] for opp in spec["opponents"])
                for seed in seeds}
        def seed_summary(rows):
            data = list(rows.values())
            se = statistics.stdev(data) / math.sqrt(len(data)) if len(data) > 1 else None
            return dict(seed_values=rows, n=len(data), mean=statistics.mean(data), standard_error=se,
                        interpretation="descriptive training-seed summary, not an independent-position confidence interval")
        endpoints[f"tilt_{tilt:g}"] = {a: {opp: seed_summary(rows) for opp, rows in panel.items()}
                                       for a, panel in values.items()}
    return dict(protocol=PROTOCOL, stage=stage, training_hours=48, lbr_evaluated=False,
        assessment_evaluated=False, strength_evidence=stage != "pilot", endpoints=endpoints,
        primary="tilt_0", lower_is_better=True,
        units="big blinds per supported audited root (not mbb per whole-game hand)",
        metric="remaining-round root response gap against a fixed opponent",
        caveats=["Diagnostic conditional on frozen positions and opponents, not exploitability or equilibrium convergence.",
                 "Pilot is correctness/runtime evidence only; development is not a locked assessment.",
                 "Eight opponents are equally weighted within each training seed before averaging seeds.",
                 "Training seeds across algorithms are independent, not paired by their integer labels.",
                 "Tilts -2/+2 are synthetic range sensitivity checks; tilt 0 is primary.",
                 "No combined score or winner is formed by mixing this audit with head-to-head winnings."])


def write_comparison(output, result, stage):
    from fhp_evaluation.best_response.runner import write_json
    summary = summarise(result, stage)
    output = Path(output)
    write_json(output / "threeway_summary.json", summary)
    parts = ["<!doctype html><meta charset='utf-8'><title>Three-way 48h strategic audit</title>",
             f"<h1>Three-way 48-hour strategic audit: {stage}</h1>",
             "<p>Lower root response gaps mean less foregone value against this fixed panel. "
             "This is not exploitability. Values are BB per audited root.</p>",
             '<p><a href="report.html">Detailed positions, local gaps, pot odds and implementation checks</a></p>',
             "<table><tr><th>Algorithm</th><th>Panel mean</th><th>Training-seed values</th><th>Seed SE</th></tr>"]
    for algorithm in ALGORITHMS:
        row = summary["endpoints"]["tilt_0"][algorithm]["equal_weight_panel"]
        parts.append(f"<tr><td>{LABELS[algorithm]}</td><td>{row['mean']:.6f}</td>"
                     f"<td>{html.escape(str(row['seed_values']))}</td><td>{row['standard_error']}</td></tr>")
    parts.append("</table><ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in summary["caveats"]) + "</ul>")
    (output / "threeway_report.html").write_text("\n".join(parts))
    return summary


def restore_archive_stamps(directory, spec):
    """Restore only staged-file metadata after a verified byte-identical re-download.

    The shared suite fingerprints mtimes as well as every chunk's SHA256.
    A new VM must retain those original stamps for its resumable board caches.
    Never alter archive contents or the immutable cloud training source.
    """
    from fhp_evaluation.loaders import sha256_file
    manifest = Path(directory) / "manifest.json"
    if not manifest.exists():
        return
    saved = read(manifest)
    if saved["spec"] != spec:
        raise ValueError("Cannot restore archive metadata for changed inputs")
    for label, policy in spec["policies"].items():
        if policy["family"] != "sd_cfr":
            continue
        root = Path(policy["checkpoint"]).parent.resolve()
        native = saved["policies"][label]["native"]
        for relative, size, stamp in native["archive_file_stamps"]:
            path = (root / relative).resolve()
            if path.parent != root or path.stat().st_size != size or sha256_file(path) != native["chunk_hashes"][relative]:
                raise ValueError("Changed archive chunk; refusing resume")
            os.utime(path, ns=(stamp, stamp))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "development"), required=True)
    parser.add_argument("--export-workers", type=int, choices=(1, 2, 4), default=4)
    for name in ("sources", "suite-repo", "sd-repo", "ucv-repo", "vr-repo", "anchor-root", "output", "execution-refs"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    import fhp_evaluation
    if Path(fhp_evaluation.__file__).resolve().parent.parent != args.suite_repo.resolve():
        raise ValueError("Use the pinned shared suite, not a vendored fhp_evaluation package")
    from fhp_evaluation.best_response.runner import write_json
    from fhp_evaluation.loaders import sha256_file
    from fhp_evaluation.strategic_audit.exp16 import LIBRARY_SEED
    from fhp_evaluation.strategic_audit.library import build_library
    from fhp_evaluation.strategic_audit.validation import validate
    from fhp_evaluation.strategic_audit.workflow import run, validate_spec
    from fhp_evaluation.strategic_audit.board_export import AuditDeferred
    source = read(args.sources)
    # Source validation ran in the isolated match-runner process before this one.
    portable = lambda rows: [{k: v for k, v in row.items() if k != "path"} for row in rows]
    if portable(source["records"]) != source["identity"]["sources"] or source["identity"].get("lbr_evaluated") is not False:
        raise ValueError("Strategic inputs do not match the validated no-LBR source identity")
    repos = dict(sd=args.sd_repo, ucv=args.ucv_repo, vr=args.vr_repo)
    refs = read(args.execution_refs)
    for name, repo in (("suite_ref", args.suite_repo), ("sd_audit_ref", args.sd_repo),
                       ("ucv_ref", args.ucv_repo), ("repo_ref", args.vr_repo)):
        actual = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                text=True, capture_output=True).stdout.strip()
        if actual != refs[name]:
            raise ValueError(f"Unpinned native/evaluator checkout: {name}")
    spec = make_spec(source["records"], repos, args.anchor_root, args.stage)
    validate_spec(spec)  # All candidate and anchor hashes checked before pickle loading.
    library = build_library(LIBRARY_SEED, pilot=args.stage == "pilot",
                            reference_deals=8 if args.stage == "pilot" else 64)
    resources = export_resources(args.stage, args.export_workers)
    contract = dict(protocol=PROTOCOL, stage=args.stage, source_identity=source["identity"],
                    execution_refs=refs, spec=spec, library=library, threads=4,
                    board_exports=resources,
                    wrapper_sha256=sha256_file(__file__), lbr_evaluated=False, assessment_evaluated=False)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists() and read(contract_path) != contract:
        raise ValueError("Changed strategic contract; choose a fresh run ID")
    write_json(contract_path, contract)
    marker = output / "analysis" / "SUCCESS.json"
    if marker.exists():
        success = read(marker)
        if success["contract_sha256"] != digest(contract) or any(
                sha256_file(output / "analysis" / p) != sha for p, sha in success["files"].items()):
            raise ValueError("Changed completed strategic report")
        return
    write_json(output / "STATUS.json", dict(status="running", stage=args.stage))
    try:
        write_json(output / "oracle_validation.json", validate())
        restore_archive_stamps(output, spec)
        result_path = run(spec, library, output, partition="development", include_reference=True,
                          unlock_assessment=False, resume=(output / "manifest.json").exists(), threads=4,
                          max_seconds=10800 if args.stage == "pilot" else 75600, max_rss_mb=52000,
                          board_exports=resources)
        write_comparison(output / "analysis", read(result_path), args.stage)
        attempts = {str(p.relative_to(output)): read(p) for p in (output / "workers").glob("**/attempt_*/run.json")}
        write_json(output / "analysis/runtime_summary.json", dict(attempts={k: {field: v.get(field)
            for field in ("status", "elapsed_seconds", "peak_rss_bytes")} for k, v in attempts.items()},
            board_exports=resources,
            export_segments=[read(p) for p in sorted((output / "export_segments").glob("attempt_*/progress.json"))],
            note="First-wave timings benchmark real full-archive throughput, not serial speedup. "
                 "Caps are safety limits; completion may require identical-code resume segments."))
        files = {p.name: sha256_file(p) for p in (output / "analysis").iterdir() if p.is_file()}
        write_json(output / "STATUS.json", dict(status="complete", stage=args.stage))
        write_json(marker, dict(status="succeeded", stage=args.stage, contract_sha256=digest(contract),
                               execution_refs=refs, assessment_evaluated=False, lbr_evaluated=False, files=files))
    except AuditDeferred as error:
        write_json(output / "STATUS.json", dict(status="deferred_time_budget", stage=args.stage,
                   message=str(error), complete=False,
                   detail="No complete comparison published. Resume this new-protocol run with identical code/settings."))
        print(f"Audit incomplete: {error}", flush=True)
    except BaseException as error:
        write_json(output / "STATUS.json", dict(status="incomplete", stage=args.stage,
                   message=str(error), detail="See status.json and workers/<policy>/<board>/attempt_*/ logs; no complete comparison published."))
        raise


if __name__ == "__main__":
    main()
