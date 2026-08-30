"""Run both approved VR-Deep variants on canonical FHP and analyse the results."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback

from fhp_vr_deep.game import serialisable_game_definition
from fhp_vr_deep.io_utils import read_json, repository_commit, write_json

from .analyse import aggregate_results, discover_results
from .config import (
    ALGORITHMS,
    APPROVAL_STATUS,
    APPROVED_CONFIG,
    APPROVED_PROTOCOL,
    CHECKPOINT_TRAINING_SECONDS,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    HEAD_TO_HEAD_DEALS_PER_PAIR,
    REFERENCE_VM,
    TRAINING_CONFIG_SHA256,
    UPSTREAM,
    smoke_config,
    validate_config,
)
from .worker import run_training_worker


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _parse_csv(value: str | None, default) -> list[str]:
    if value is None:
        return [str(item) for item in default]
    selected = [item.strip() for item in value.split(",") if item.strip()]
    if not selected:
        raise ValueError("At least one value is required")
    return selected


def _parse_algorithms(value: str | None) -> list[str]:
    selected = _parse_csv(value, ALGORITHMS)
    unknown = sorted(set(selected) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown algorithm ids: {', '.join(unknown)}")
    return selected


def _parse_seeds(value: str | None) -> list[int]:
    return [int(item) for item in _parse_csv(value, DEFAULT_SEEDS)]


def _worker_mode(input_path: Path, output_path: Path) -> int:
    payload = read_json(input_path)
    result = run_training_worker(payload)
    write_json(output_path, result)
    return 0


def _run_subprocess(run_dir: Path, payload: dict[str, object]) -> dict[str, object]:
    algorithm_id = str(payload["algorithm_id"])
    seed = int(payload["seed"])
    stem = f"{algorithm_id}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(input_path, payload)
    command = [
        sys.executable,
        "-m",
        "experiments.fhp.exp1_leduc_config_transfer.run",
        "--worker-input-json",
        str(input_path),
        "--worker-output-json",
        str(output_path),
    ]
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if completed.returncode:
        raise RuntimeError(f"{stem} failed; see {log_path}")
    result = read_json(output_path)
    result["_resolved_worker_dir"] = str(Path(str(payload["worker_dir"])).resolve())
    return result


def _aggregate_existing(args) -> int:
    results = discover_results(args.aggregate_run_dir)
    if not results:
        raise ValueError("No worker_runs/*/result.json files were found")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{EXPERIMENT_NAME}_aggregated_{timestamp}"
    deals = args.head_to_head_deals or HEAD_TO_HEAD_DEALS_PER_PAIR
    aggregate_results(
        results,
        output_dir,
        head_to_head_deals=deals,
        run_evaluation=not args.skip_head_to_head,
    )
    print(output_dir.resolve())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", help="comma-separated training seeds")
    parser.add_argument("--algorithms", help="comma-separated algorithm ids")
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        nargs=2,
        metavar=("FIRST", "FINAL"),
    )
    parser.add_argument("--head-to-head-deals", type=int)
    parser.add_argument("--skip-head-to-head", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--aggregate-run-dir",
        type=Path,
        action="append",
        help="aggregate one or more existing Experiment 1 run directories",
    )
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both internal worker paths are required")
        return _worker_mode(args.worker_input_json, args.worker_output_json)
    if args.aggregate_run_dir:
        return _aggregate_existing(args)
    if APPROVAL_STATUS != "approved":
        raise RuntimeError("Experiment 1 configuration has not been approved")

    seeds = _parse_seeds(args.seeds)
    algorithms = _parse_algorithms(args.algorithms)
    is_smoke = bool(args.smoke)
    config = smoke_config() if is_smoke else deepcopy(APPROVED_CONFIG)
    validate_config(config, production=not is_smoke)
    checkpoint_seconds = tuple(
        args.checkpoint_seconds
        or ((1e-6, 2e-6) if is_smoke else CHECKPOINT_TRAINING_SECONDS)
    )
    if len(checkpoint_seconds) != 2 or checkpoint_seconds[0] >= checkpoint_seconds[1]:
        raise ValueError("Checkpoint seconds must contain two increasing values")
    head_to_head_deals = int(
        args.head_to_head_deals
        if args.head_to_head_deals is not None
        else (4 if is_smoke else HEAD_TO_HEAD_DEALS_PER_PAIR)
    )
    if head_to_head_deals <= 0:
        raise ValueError("Head-to-head deal count must be positive")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{EXPERIMENT_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "approval_status": APPROVAL_STATUS,
        "is_smoke": is_smoke,
        "algorithms": {key: ALGORITHMS[key] for key in algorithms},
        "seeds": seeds,
        "game": serialisable_game_definition(),
        "training_config": config,
        "approved_training_config_sha256": TRAINING_CONFIG_SHA256,
        "checkpoint_training_seconds": list(checkpoint_seconds),
        "head_to_head_deals_per_pair": head_to_head_deals,
        "protocol": APPROVED_PROTOCOL,
        "reference_vm": REFERENCE_VM,
        "upstream": UPSTREAM,
        "repository_commit": repository_commit(REPOSITORY_ROOT),
        "node_counter_scope": "training traversal nodes; evaluation games excluded",
        "training_clock_scope": (
            "solver time excluding checkpoint policy fitting, serialization, and evaluation"
        ),
        "upstream_correctness_corrections": [
            "Immediate-regret optimizer is attached to the immediate-regret model.",
            "PDCFR trainer arguments use their declared order.",
            "Immediate-regret reinitialization is independent of cumulative reset.",
            "Circular history-value replay occupancy is tracked separately from write index.",
        ],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "experiment_metadata.json", metadata)

    results = []
    failures = []
    stop_requested = False
    for algorithm_id in algorithms:
        for seed in seeds:
            worker_dir = run_dir / "worker_runs" / f"{algorithm_id}_seed_{seed}"
            payload = {
                "algorithm_id": algorithm_id,
                "seed": seed,
                "config": config,
                "checkpoint_seconds": checkpoint_seconds,
                "is_smoke": is_smoke,
                "worker_dir": str(worker_dir.resolve()),
            }
            try:
                LOGGER.info("Running %s seed %s", algorithm_id, seed)
                results.append(_run_subprocess(run_dir, payload))
                write_json(run_dir / "partial_results.json", results)
            except Exception as exc:  # pragma: no cover - operational path
                failure = {
                    "algorithm_id": algorithm_id,
                    "seed": seed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                write_json(run_dir / "failed_runs.json", failures)
                LOGGER.error("%s seed %s failed: %s", algorithm_id, seed, exc)
                if not args.continue_on_error:
                    stop_requested = True
                    break
        if stop_requested:
            break

    if results:
        aggregate_results(
            results,
            run_dir,
            head_to_head_deals=head_to_head_deals,
            run_evaluation=not args.skip_head_to_head,
        )
    write_json(
        run_dir / "run_status.json",
        {
            "completed_workers": len(results),
            "expected_workers": len(algorithms) * len(seeds),
            "failures": failures,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
