"""Fresh-process, matched-workload profiling. No evaluation or training launch."""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

MODULE = "experiments.fhp.exp4_vr_deep_thread_profile.run"
ROOT = Path(__file__).resolve().parents[3]


def thread_environment(threads):
    env = os.environ.copy()
    env.update({key: str(threads) for key in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="",
               OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    return env


def read(path):
    return json.loads(Path(path).read_text())


def configure_threads(threads):
    import torch
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != str(threads):
            raise ValueError(f"{key} must be set before starting the child interpreter")
    torch.set_num_threads(threads)
    # One inter-op scheduler for every arm; tested variable is intra-op/BLAS.
    torch.set_num_interop_threads(1)


def runtime():
    import numpy as np
    import torch
    import psutil
    from fhp_vr_deep.io_utils import repository_commit
    return dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                torch_configuration=torch.__config__.show(), platform=platform.platform(),
                cpu_count=os.cpu_count(), physical_cpu_count=psutil.cpu_count(logical=False),
                cpu_affinity=psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None,
                cpu_model=Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else platform.processor(),
                memory_bytes=psutil.virtual_memory().total,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                thread_environment={key: os.environ.get(key) for key in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_DYNAMIC", "MKL_DYNAMIC")},
                commit=repository_commit(ROOT))


def prepare(scratch, output, spec):
    import torch
    from experiments.fhp.exp1_vr_deep_pdcfr_24h.train import write_json
    from fhp_vr_deep.io_utils import sha256_file
    from . import workload as w

    w.kernel_warmup(spec)
    solver = w.new_solver(spec)
    warmups = []
    for index in range(spec["warmup_iterations"]):
        row = w.measure(solver, solver.iteration)
        warmups.append(row)
        write_json(output / "preparation_progress.json", dict(completed_warmup_iterations=index + 1, timings=warmups))
        print(json.dumps(dict(event="fixture_warmup", iteration=index + 1, seconds=row["seconds"])), flush=True)
    w.clear_features(solver)
    fixtures = {"whole": w.save_fixture(scratch / "whole.pt", solver, w.trainers(solver))}
    reference = {}

    def capture(component, fn, buffers):
        fixtures[component] = w.save_fixture(scratch / f"{component}.pt", solver, buffers)
        reference[component] = w.measure(solver, fn)
        torch.save(w.endpoint(solver), scratch / f"{component}_endpoint.pt")
        return None

    regret, critic = solver.train_regret, solver.train_baseline
    solver.train_regret = lambda p: capture(f"regret_{p}", lambda: regret(p), [f"regret_{p}"])
    solver.train_baseline = lambda p: capture(f"critic_{p}", lambda: critic(p), ["critic"])
    trace, inference = w.ChoiceTrace(), w.InferenceTrace()
    with trace.observe(), inference.observe(solver):
        reference["whole"] = w.measure(solver, solver.iteration)
    reference["whole"]["final_replay_sha256"] = w.replay_digest(solver)
    del solver.train_regret, solver.train_baseline
    torch.save(w.endpoint(solver), scratch / "whole_endpoint.pt")
    capture("policy", solver.train_average_policy, ["policy"])
    # Production fitting must not silently skip work because replay is too small.
    cfg = spec["training_config"]
    expected_calls = dict(regret_0=cfg["advantage_network_train_steps"], regret_1=cfg["advantage_network_train_steps"],
                          critic_0=cfg["baseline_network_train_steps"] + 1,
                          critic_1=cfg["baseline_network_train_steps"] + 1,
                          policy=cfg["ave_policy_network_train_steps"])
    expected_calls["whole"] = sum(value for name, value in expected_calls.items() if name != "policy")
    for name, row in reference.items():
        if row["sampler"]["calls"] != expected_calls[name]:
            raise ValueError(f"Incomplete fitting workload: {name}")
    hashes = {name: sha256_file(scratch / f"{name}.pt") for name in fixtures}
    write_json(scratch / "trace.json", trace.events)
    write_json(scratch / "inference.json", inference.events)
    write_json(scratch / "reference.json", reference)
    write_json(output / "fixture_manifest.json", dict(
        spec=spec, fixture_sha256=hashes, buffers=fixtures, warmup_timings=warmups,
        reference_sampler={name: row["sampler"] for name, row in reference.items()},
        reference_nodes=reference["whole"]["nodes"], choice_events=len(trace.events),
        choice_trace_sha256=sha256_file(scratch / "trace.json"), runtime=runtime(),
        inference_trace_sha256=sha256_file(scratch / "inference.json"), inference_events=len(inference.events),
        note="Real early-training replay; occupancy is measured, not claimed to be mature/full. "
             "Reference timings include fixture capture overhead: never use them as performance measurements."))


def trial(scratch, output, spec, threads, repeat):
    import torch
    from experiments.fhp.exp1_vr_deep_pdcfr_24h.train import write_json
    from fhp_vr_deep.io_utils import sha256_file
    from . import workload as w
    from .config import COMPONENTS

    reference = read(scratch / "reference.json")
    manifest = read(output / "fixture_manifest.json")
    w.kernel_warmup(spec)
    gc.collect()
    components = list(COMPONENTS)
    if repeat % 2:
        components = ["whole", *components[:-1]]
    result = dict(threads=threads, repeat=repeat, runtime=runtime(), component_order=components, measurements={})
    for component in components:
        path = scratch / f"{component}.pt"
        if sha256_file(path) != manifest["fixture_sha256"][component]:
            raise ValueError(f"Fixture changed: {component}")
        solver = w.load_fixture(path, spec)
        trace = w.ChoiceTrace(read(scratch / "trace.json")) if component == "whole" else None
        if trace is not None:
            for filename, key in (("trace.json", "choice_trace_sha256"), ("inference.json", "inference_trace_sha256")):
                if sha256_file(scratch / filename) != manifest[key]:
                    raise ValueError(f"Trace changed: {filename}")
            inference = w.InferenceTrace(read(scratch / "inference.json"))
            with trace.observe(), inference.observe(solver):
                row = w.measure(solver, solver.iteration)
            row.update(choice_events=trace.cursor, choice_replay_divergences=trace.divergences,
                       inference_events=inference.cursor, inference_different_outputs=inference.different_outputs,
                       inference_max_abs_difference=inference.max_abs_difference)
            if trace.divergences:
                raise ValueError("Different choices despite identical collection outputs/RNG")
            row["final_replay_sha256"] = w.replay_digest(solver)
            if row["final_replay_sha256"] != reference[component]["final_replay_sha256"]:
                raise ValueError("Whole-iteration numerical replay differs from reference")
        else:
            row = w.measure(solver, w.fit_call(solver, component))
        if row["sampler"] != reference[component]["sampler"]:
            raise ValueError(f"Minibatches differ from reference: {component}")
        if any(row[key] != reference[component][key] for key in ("nodes", "traversals", "iterations")):
            raise ValueError(f"Iteration workload differs from reference: {component}")
        expected = torch.load(scratch / f"{component}_endpoint.pt", weights_only=True)
        row["endpoint_max_abs_difference_from_8_threads"] = w.endpoint_difference(solver, expected)
        row["fixture_sha256"] = manifest["fixture_sha256"][component]
        row["workload_verified"] = True
        result["measurements"][component] = row
        write_json(output / "trials" / f"repeat_{repeat}_threads_{threads}.json", result)
        print(json.dumps(dict(event="measurement", repeat=repeat, threads=threads, component=component,
                              seconds=row["seconds"])), flush=True)
        del solver, expected
        gc.collect()
    return result


def summarise(output, spec):
    from experiments.fhp.exp1_vr_deep_pdcfr_24h.train import write_json
    from fhp_vr_deep.io_utils import write_csv
    from .config import COMPONENTS, THREAD_COUNTS

    results = [read(output / "trials" / f"repeat_{repeat}_threads_{threads}.json")
               for repeat in range(spec["repeats"]) for threads in THREAD_COUNTS]
    raw = []
    for result in results:
        if set(result["measurements"]) != set(COMPONENTS):
            raise ValueError("Incomplete trial cannot be summarised")
        for component, row in result["measurements"].items():
            if not row["workload_verified"]:
                raise ValueError("Unverified workload")
            raw.append(dict(threads=result["threads"], repeat=result["repeat"], component=component,
                            **{key: value for key, value in row.items() if not isinstance(value, dict)}))
        # Active fitting excludes checkpoint-only average-policy fitting.
        raw.append(dict(threads=result["threads"], repeat=result["repeat"], component="active_fitting",
                        seconds=sum(result["measurements"][name]["seconds"] for name in COMPONENTS[:4])))
    summary = []
    for component in (*COMPONENTS, "active_fitting"):
        baseline = {row["repeat"]: row["seconds"] for row in raw
                    if row["component"] == component and row["threads"] == 8}
        for threads in THREAD_COUNTS:
            rows = [row for row in raw if row["component"] == component and row["threads"] == threads]
            values = [row["seconds"] for row in rows]
            summary.append(dict(component=component, threads=threads, repeats=len(values),
                                median_seconds=statistics.median(values), min_seconds=min(values), max_seconds=max(values),
                                paired_median_speedup_vs_8=statistics.median(
                                    baseline[row["repeat"]] / row["seconds"] for row in rows)))
    fastest = min((row for row in summary if row["component"] == "whole"), key=lambda row: row["median_seconds"])
    report = dict(is_smoke=spec["is_smoke"], timings=summary,
                  fastest_observed_whole_iteration_threads=None if spec["is_smoke"] else fastest["threads"],
                  interpretation="Technical repeats on one early-training fixture, not independent learning seeds. "
                  "Whole iterations replay reference game paths and collection outputs while executing all inference; "
                  "fitting fixtures have identical starting models, "
                  "optimizer state, replay and minibatches. No claim of policy quality or convergence. "
                  "Small timing differences need confirmation; no configuration is automatically changed.")
    write_csv(output / "analysis" / "measurements.csv", raw)
    write_csv(output / "analysis" / "timings.csv", summary)
    write_json(output / "analysis" / "summary.json", report)
    return report


def orchestrate(output, *, smoke=False, remote_uri=None, scratch_root=None):
    from experiments.fhp.exp1_vr_deep_pdcfr_24h.train import write_json, publish
    from .config import protocol, trial_order, PROFILE_MAX_SECONDS

    output.mkdir(parents=True, exist_ok=False)
    spec = protocol(smoke)
    write_json(output / "protocol.json", spec)
    deadline = time.monotonic() + (1800 if smoke else PROFILE_MAX_SECONDS)
    order = trial_order(spec["repeats"])
    write_json(output / "trial_order.json", order)

    def child(action, scratch, threads, repeat=0):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Profiling wall-time budget exhausted; partial outputs retained")
        log = output / "logs" / f"{action}_{repeat}_{threads}.txt"
        log.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-m", MODULE, action, "--scratch", str(scratch),
                   "--output-root", str(output), "--threads", str(threads), "--repeat", str(repeat)]
        print(json.dumps(dict(event="start_child", action=action, threads=threads, repeat=repeat)), flush=True)
        try:
            with log.open("w") as stream:
                subprocess.run(command, check=True, timeout=remaining, env=thread_environment(threads),
                               stdout=stream, stderr=subprocess.STDOUT)
        except Exception:
            print(log.read_text()[-8000:], flush=True)
            raise
        publish(output, remote_uri)

    try:
        # Large temporary fixtures remain on the VM's disk; never upload replay.
        with tempfile.TemporaryDirectory(prefix="vr-exp4-fixtures-", dir=scratch_root) as temporary:
            scratch = Path(temporary)
            child("prepare", scratch, 8)
            for repeat, threads in order:
                child("trial", scratch, threads, repeat)
            report = summarise(output, spec)
        write_json(output / "SUCCESS.json", dict(complete=True, is_smoke=smoke,
                                                trials=len(order), workload_verified=True))
        # Batch's finalisation trap uploads diagnostics and publishes markers last.
        # Local runs retain the marker immediately, but child-stage sync excludes it.
        publish(output, remote_uri)
        return report
    except BaseException:
        write_json(output / "FAILURE.json", dict(traceback=traceback.format_exc(), complete=False))
        # A failed upload must never leave a local success marker for a shell trap.
        (output / "SUCCESS.json").unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "smoke", "prepare", "trial"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--remote-uri")
    args = parser.parse_args(argv)
    if args.action in ("run", "smoke"):
        orchestrate(args.output_root, smoke=args.action == "smoke", remote_uri=args.remote_uri,
                    scratch_root=args.scratch_root)
    else:
        if args.scratch is None:
            parser.error("Internal worker requires --scratch")
        configure_threads(args.threads)
        spec = read(args.output_root / "protocol.json")
        if args.action == "prepare":
            prepare(args.scratch, args.output_root, spec)
        else:
            trial(args.scratch, args.output_root, spec, args.threads, args.repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
