"""Single-algorithm, single-seed training worker for FHP Archived Experiment 1."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import time
import traceback
from typing import Mapping

import numpy as np
import psutil

from fhp_vr_deep.game import load_fhp_game, serialisable_game_definition
from fhp_vr_deep.io_utils import (
    canonical_sha256,
    peak_rss_mib,
    repository_commit,
    sha256_file,
    write_csv,
    write_json,
)
from vr_deep_cfr import VRDeepDCFRPlus, VRDeepPDCFRPlus
from vr_deep_cfr.logger import Logger
from vr_deep_cfr.policy_snapshots import (
    LoadedVRPolicy,
    load_policy_snapshot_payload,
    save_policy_snapshot,
)

from .config import (
    ALGORITHMS,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    REFERENCE_VM,
    TRAINING_CONFIG_SHA256,
    UPSTREAM,
    validate_config,
)


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _solver_kwargs(
    algorithm_id: str,
    seed: int,
    config: Mapping[str, object],
) -> tuple[type, dict]:
    spec = ALGORITHMS[algorithm_id]
    control_fields = {"max_num_iterations", "preserve_evaluation_rng"}
    kwargs = {
        key: value
        for key, value in deepcopy(dict(config)).items()
        if key not in control_fields
    }
    kwargs.update(
        {
            "num_episodes": (
                2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
            ),
            "alpha": float(spec["alpha"]),
            "gamma": float(spec["gamma"]),
            "seed": int(seed),
            "logger": Logger(verbose=True),
        }
    )
    if spec["reinitialize_imm_regret_networks"] is not None:
        kwargs["reinitialize_imm_regret_networks"] = bool(
            spec["reinitialize_imm_regret_networks"]
        )
    solver_class = {
        "VRDeepDCFRPlus": VRDeepDCFRPlus,
        "VRDeepPDCFRPlus": VRDeepPDCFRPlus,
    }[spec["class_name"]]
    return solver_class, kwargs


def _decorate_curve(raw: Mapping[str, object], algorithm_id: str, seed: int, index: int):
    spec = ALGORITHMS[algorithm_id]
    row = dict(raw)
    row.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "algorithm_id": algorithm_id,
            "algorithm_label": spec["algorithm_label"],
            "seed": int(seed),
            "checkpoint_index": int(index),
            "outer_iteration": int(raw.get("iteration", 0)),
        }
    )
    return row


def run_training_worker(payload: Mapping[str, object]) -> dict[str, object]:
    worker_started = time.perf_counter()
    algorithm_id = str(payload["algorithm_id"])
    if algorithm_id not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm id: {algorithm_id}")
    seed = int(payload["seed"])
    config = dict(payload["config"])
    is_smoke = bool(payload.get("is_smoke", False))
    validate_config(config, production=not is_smoke)
    checkpoint_seconds = tuple(float(value) for value in payload["checkpoint_seconds"])
    if len(checkpoint_seconds) != 2 or checkpoint_seconds[0] >= checkpoint_seconds[1]:
        raise ValueError("Archived Experiment 1 requires exactly two increasing time checkpoints")
    worker_dir = Path(str(payload["worker_dir"]))
    worker_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = worker_dir / "checkpoints"
    checkpoints_dir.mkdir()
    spec = ALGORITHMS[algorithm_id]

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "algorithm_id": algorithm_id,
        "algorithm_label": spec["algorithm_label"],
        "seed": seed,
        "is_smoke": is_smoke,
        "checkpoint_training_seconds": list(checkpoint_seconds),
        "stopping_rule": "final_training_time_checkpoint",
        "game": serialisable_game_definition(),
        "training_config": config,
        "training_config_sha256": canonical_sha256(config),
        "approved_production_config_sha256": TRAINING_CONFIG_SHA256,
        "algorithm_parameters": {
            "alpha": spec["alpha"],
            "gamma": spec["gamma"],
            "reinitialize_imm_regret_networks": spec[
                "reinitialize_imm_regret_networks"
            ],
        },
        "upstream": UPSTREAM,
        "repository_commit": repository_commit(REPOSITORY_ROOT),
        "reference_vm": REFERENCE_VM,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(worker_dir / "run_manifest.json", manifest)

    solver_class, kwargs = _solver_kwargs(algorithm_id, seed, config)
    try:
        solver = solver_class(**kwargs)
    except BaseException as exc:
        write_json(
            worker_dir / "failure.json",
            {
                "phase": "solver_initialization",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "peak_rss_mib": peak_rss_mib(),
            },
        )
        raise

    solver.target_nodes_touched = None
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.max_wall_clock_seconds = None
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = False
    solver.early_evaluation_node_thresholds = ()
    solver.training_time_checkpoint_seconds = checkpoint_seconds
    solver.stop_after_final_training_time_checkpoint = True
    snapshot_rows = []

    def capture_time_checkpoint(active_solver, raw_checkpoint):
        target_seconds = raw_checkpoint.get("checkpoint_target_seconds")
        if target_seconds is None:
            return
        index = len(snapshot_rows)
        filename = (
            f"{algorithm_id}_seed_{seed}_checkpoint_{index:02d}_"
            f"{float(target_seconds):g}s.pt"
        )
        snapshot_path = save_policy_snapshot(
            active_solver,
            checkpoints_dir / filename,
            algorithm_id=algorithm_id,
            algorithm_label=spec["algorithm_label"],
            seed=seed,
            config=config,
            checkpoint_row=dict(raw_checkpoint),
        )
        loaded = load_policy_snapshot_payload(snapshot_path)
        if int(loaded["nodes_touched"]) != int(active_solver.nodes_touched):
            raise RuntimeError("Saved snapshot does not match the solver node count")
        snapshot_rows.append(
            {
                "checkpoint_index": index,
                "checkpoint_kind": raw_checkpoint["checkpoint_kind"],
                "checkpoint_target_seconds": float(target_seconds),
                "training_elapsed_seconds": float(
                    raw_checkpoint["training_elapsed_seconds"]
                ),
                "wall_clock_seconds": float(raw_checkpoint["wall_clock_seconds"]),
                "nodes_touched": int(active_solver.nodes_touched),
                "outer_iteration": int(active_solver.num_iteration),
                "episode": int(active_solver.episode),
                "snapshot_path": str(snapshot_path.relative_to(worker_dir)),
                "snapshot_sha256": sha256_file(snapshot_path),
                "snapshot_size_bytes": int(snapshot_path.stat().st_size),
            }
        )
        write_json(worker_dir / "checkpoint_manifest.json", snapshot_rows)

    try:
        raw_rows = solver.solve(post_checkpoint_callback=capture_time_checkpoint)
        captured_targets = tuple(
            row["checkpoint_target_seconds"] for row in snapshot_rows
        )
        if captured_targets != checkpoint_seconds:
            raise RuntimeError(
                f"Captured time checkpoints {captured_targets}, expected {checkpoint_seconds}"
            )
        if solver.stop_reason != "training_time_budget":
            raise RuntimeError(
                f"Training stopped for {solver.stop_reason!r}, not the time budget"
            )
        curves = [
            _decorate_curve(row, algorithm_id, seed, index)
            for index, row in enumerate(raw_rows)
        ]
        final_source = worker_dir / snapshot_rows[-1]["snapshot_path"]
        final_snapshot = worker_dir / "final_policy_snapshot.pt"
        shutil.copyfile(final_source, final_snapshot)
        final_payload = load_policy_snapshot_payload(final_snapshot)
        game = load_fhp_game()
        policy = LoadedVRPolicy(game, final_snapshot)
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        probabilities = policy.action_probabilities(state)
        if not np.isclose(sum(probabilities.values()), 1.0):
            raise RuntimeError("Reloaded final policy probabilities do not sum to one")

        time_rows = [
            row for row in curves if row.get("checkpoint_target_seconds") is not None
        ]
        final_row = time_rows[-1]
        training_seconds = float(final_row["training_elapsed_seconds"])
        process = psutil.Process()
        summary = {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "algorithm_id": algorithm_id,
            "algorithm_label": spec["algorithm_label"],
            "seed": seed,
            "is_smoke": is_smoke,
            "stop_reason": solver.stop_reason,
            "final_nodes_touched": int(solver.nodes_touched),
            "final_outer_iteration": int(solver.num_iteration),
            "final_episode": int(solver.episode),
            "final_training_elapsed_seconds": training_seconds,
            "final_wall_clock_seconds": float(final_row["wall_clock_seconds"]),
            "total_worker_wall_clock_seconds": time.perf_counter() - worker_started,
            "nodes_per_training_second": (
                float(solver.nodes_touched) / training_seconds
                if training_seconds > 0.0
                else float("nan")
            ),
            "checkpoint_count": len(snapshot_rows),
            "diagnostic_checkpoint_count": len(curves),
            "peak_rss_mib": peak_rss_mib(),
            "current_rss_mib": process.memory_info().rss / (1024.0 * 1024.0),
            "final_policy_snapshot": "final_policy_snapshot.pt",
            "final_policy_snapshot_sha256": sha256_file(final_snapshot),
            "final_snapshot_nodes": int(final_payload["nodes_touched"]),
            "final_average_policy_buffer_size": len(solver.ave_policy_trainer.buffer),
            "final_regret_buffer_size_player_0": len(solver.regret_trainers[0].buffer),
            "final_regret_buffer_size_player_1": len(solver.regret_trainers[1].buffer),
            "final_history_value_buffer_size": len(solver.q_value_trainer.buffer),
        }
        result = {
            "worker_run_dir": str(worker_dir),
            "summary": summary,
            "curves": curves,
            "snapshots": snapshot_rows,
        }
        write_json(worker_dir / "checkpoint_rows.json", curves)
        write_csv(worker_dir / "checkpoint_rows.csv", curves)
        write_csv(worker_dir / "checkpoint_manifest.csv", snapshot_rows)
        write_json(worker_dir / "summary.json", summary)
        write_json(worker_dir / "result.json", result)
        return result
    except BaseException as exc:
        write_json(
            worker_dir / "failure.json",
            {
                "phase": "training_or_artifact_validation",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "nodes_touched": int(getattr(solver, "nodes_touched", 0)),
                "outer_iteration": int(getattr(solver, "num_iteration", 0)),
                "peak_rss_mib": peak_rss_mib(),
            },
        )
        raise
    finally:
        close = getattr(solver, "close", None)
        if callable(close):
            close()
