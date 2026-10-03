"""Completed-iteration time checkpoints without altering VR-Deep update rules."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
import torch

from fhp_vr_deep.game import load_fhp_game, serialisable_game_definition
from fhp_vr_deep.io_utils import (canonical_sha256, json_default, peak_rss_mib,
                                 repository_commit, sha256_file, write_csv)
from vr_deep_cfr import VRDeepPDCFRPlus
from vr_deep_cfr.logger import Logger
from vr_deep_cfr.policy_snapshots import LoadedVRPolicy, save_policy_snapshot
from . import config as default_experiment
from .config import THREADS

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, payload):
    """Commit metadata atomically; upload ignores temporary files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=json_default, allow_nan=True) + "\n")
    temporary.replace(path)


def publish(worker_dir, remote_uri, *, success=False):
    if not remote_uri:
        return
    if not remote_uri.startswith("gs://"):
        raise ValueError("Remote worker destination must be a gs:// URI")
    # Marker last: a visible SUCCESS.json certifies all referenced files landed.
    subprocess.run(["gcloud", "storage", "rsync", "--recursive",
                    "--exclude", r"(^|/)SUCCESS\.json$|\.tmp(/|$)",
                    str(worker_dir), remote_uri], check=True)
    if success:
        subprocess.run(["gcloud", "storage", "cp", str(worker_dir / "SUCCESS.json"),
                        remote_uri.rstrip("/") + "/SUCCESS.json"], check=True)


class TimedVRDeepPDCFRPlus(VRDeepPDCFRPlus):
    """Only observation/stopping changes; preserve the base iteration body."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.at_iteration_boundary = False
        self.iteration_callback = None
        self.phase_seconds = {}

    def _timed(self, name, fn, *args):
        start = time.perf_counter()
        result = fn(*args)
        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + time.perf_counter() - start
        return result

    def collect_training_data(self, player):
        return self._timed("collection", super().collect_training_data, player)

    def train_regret(self, player):
        return self._timed("regret_fitting", super().train_regret, player)

    def train_baseline(self, player):
        return self._timed("critic_fitting", super().train_baseline, player)

    def train_average_policy(self):
        return self._timed("average_policy_fitting", super().train_average_policy)

    def _maybe_run_training_time_checkpoint(self):
        if self.at_iteration_boundary:
            super()._maybe_run_training_time_checkpoint()

    def iteration(self):
        super().iteration()
        self.at_iteration_boundary = True
        try:
            if self.iteration_callback:
                self.iteration_callback(self)
            self._maybe_run_training_time_checkpoint()
        finally:
            self.at_iteration_boundary = False


def make_solver(seed, config, *, solver_class=TimedVRDeepPDCFRPlus):
    kwargs = {k: v for k, v in config.items()
              if k not in ("max_num_iterations", "preserve_evaluation_rng")}
    kwargs.update(seed=seed, logger=Logger(verbose=True),
                  num_episodes=2 * config["num_traversals"] * config["max_num_iterations"])
    solver = solver_class(**kwargs)
    solver.max_num_iterations = config["max_num_iterations"]
    solver.preserve_evaluation_rng = config["preserve_evaluation_rng"]
    solver.stop_after_final_training_time_checkpoint = True
    return solver


def verify_policy(path, solver=None):
    policy = LoadedVRPolicy(load_fhp_game(), path)
    state = policy.game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    probs = policy.action_probabilities(state)
    if not probs or not np.isclose(sum(probs.values()), 1.0):
        raise ValueError("Reloaded policy is not a valid distribution")
    if solver is not None:
        expected = solver.ave_policy_trainer.action_probabilities(state)
        if any(not np.isclose(probs[a], expected[a], rtol=1e-6, atol=1e-7) for a in probs):
            raise ValueError("Saved policy does not reproduce the fitted policy")
    return policy.metadata


def run_worker(output_root, task_index, *, smoke=False, remote_uri=None, threads=THREADS,
               experiment=default_experiment, solver_factory=None, after_training=None):
    ALGORITHM_ID, ALGORITHM_LABEL = experiment.ALGORITHM_ID, experiment.ALGORITHM_LABEL
    SEEDS, schedule, task_name = experiment.SEEDS, experiment.schedule, experiment.task_name
    if smoke and task_index != 0:
        raise ValueError("Smoke uses seed 0 only")
    if threads < 1 or (not smoke and threads != THREADS):
        raise ValueError(f"Production uses {THREADS} Torch CPU threads")
    if remote_uri and not remote_uri.startswith("gs://"):
        raise ValueError("Remote worker destination must start with gs://")
    name = task_name(task_index)
    seed = SEEDS[task_index]
    worker_dir = Path(output_root) / "workers" / name
    # No silent overwrite/restart: policy snapshots are not resumable states.
    worker_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    spec = experiment.contract(smoke)
    config = spec["training_config"]
    manifest = dict(spec, seed=seed, task_index=task_index, game=serialisable_game_definition(),
                    repository_commit=repository_commit(ROOT), torch_threads=torch.get_num_threads(),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                 platform=platform.platform()))
    write_json(worker_dir / "run_manifest.json", manifest)
    started = time.perf_counter()
    solver = None
    snapshots, curves = [], []
    try:
        solver = (solver_factory or make_solver)(seed, config)
        solver.training_time_checkpoint_seconds = tuple(r["checkpoint_target_seconds"] for r in schedule(smoke))

        def progress(active):
            row = dict(active.logger._pending)
            row.update(seed=seed, algorithm_id=ALGORITHM_ID, iteration=active.num_iteration,
                       nodes_touched=active.nodes_touched, episode=active.episode,
                       training_elapsed_seconds=active._training_elapsed_seconds(),
                       wall_clock_seconds=active._wall_clock_seconds(), peak_rss_mib=peak_rss_mib(),
                       rss_mib=psutil.Process().memory_info().rss / 2**20,
                       phase_seconds=dict(active.phase_seconds))
            with (worker_dir / "training_progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row, default=json_default) + "\n")
            print(json.dumps({k: row[k] for k in ("seed", "iteration", "nodes_touched", "training_elapsed_seconds", "rss_mib")}), flush=True)

        def checkpoint(active, raw):
            index = len(snapshots)
            target = schedule(smoke)[index]
            if raw["checkpoint_target_seconds"] != target["checkpoint_target_seconds"]:
                raise ValueError("Unexpected checkpoint threshold")
            row = dict(raw, **{k: v for k, v in target.items() if k not in raw})
            row.update(seed=seed, algorithm_id=ALGORITHM_ID, checkpoint_index=index,
                       completed_iteration=active.num_iteration,
                       actual_training_elapsed_seconds=raw["training_elapsed_seconds"],
                       overshoot_seconds=max(0, raw["training_elapsed_seconds"] - target["checkpoint_target_seconds"]))
            path = worker_dir / "checkpoints" / f"{ALGORITHM_ID}_seed_{seed}_{target['checkpoint_id']}.pt"
            temporary = path.with_name(path.name + ".tmp")
            save_policy_snapshot(active, temporary, algorithm_id=ALGORITHM_ID,
                                 algorithm_label=ALGORITHM_LABEL, seed=seed, config=config,
                                 checkpoint_row=row)
            temporary.replace(path)
            verify_policy(path, active)
            snapshots.append(dict(row, path=str(path.relative_to(worker_dir)),
                                  sha256=sha256_file(path), size_bytes=path.stat().st_size))
            curves.append(row)
            write_json(worker_dir / "checkpoint_manifest.json", snapshots)
            write_csv(worker_dir / "checkpoint_manifest.csv", snapshots)
            write_json(worker_dir / "checkpoint_rows.json", curves)
            write_csv(worker_dir / "checkpoint_rows.csv", curves)
            publish(worker_dir, remote_uri)

        solver.iteration_callback = progress
        solver.solve(post_checkpoint_callback=checkpoint)
        if len(snapshots) != len(schedule(smoke)) or solver.stop_reason != "training_time_budget":
            raise RuntimeError(f"Incomplete training: {len(snapshots)} checkpoints; {solver.stop_reason}")
        final = snapshots[-1]
        summary = dict(experiment_name=spec["experiment_name"], algorithm_id=ALGORITHM_ID, seed=seed,
                       is_smoke=smoke, stop_reason=solver.stop_reason, checkpoint_count=len(snapshots),
                       final_nodes_touched=solver.nodes_touched, final_outer_iteration=solver.num_iteration,
                       final_training_elapsed_seconds=final["training_elapsed_seconds"],
                       final_wall_clock_seconds=solver._wall_clock_seconds(),
                       total_worker_wall_clock_seconds=time.perf_counter() - started,
                       nodes_per_training_second=solver.nodes_touched / final["training_elapsed_seconds"],
                       peak_rss_mib=peak_rss_mib(), phase_seconds=solver.phase_seconds,
                       final_policy_snapshot=final["path"], final_policy_sha256=final["sha256"],
                       buffers={}, model_parameters={}, input_sizes={})
        for name, trainer in (("policy", solver.ave_policy_trainer), ("critic", solver.q_value_trainer),
                              ("regret_0", solver.regret_trainers[0]), ("regret_1", solver.regret_trainers[1])):
            buf = trainer.buffer
            summary["buffers"][name] = dict(occupancy=min(len(buf), buf.buffer_size),
                                           seen_or_write_index=buf.cur_id, capacity=buf.buffer_size,
                                           array_bytes=sum(a.nbytes for a in vars(buf).values()
                                                           if isinstance(a, np.ndarray)))
            summary["model_parameters"][name] = sum(p.numel() for p in trainer.model.parameters())
            summary["input_sizes"][name] = trainer.input_size
        if after_training is not None:
            # Run only after solve returns: checkpoint RNG has been restored,
            # the final complete iteration has finished and Ray has shut down.
            summary.update(after_training(solver, worker_dir, final, manifest))
        write_json(worker_dir / "summary.json", summary)
        files = [p for p in worker_dir.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
        success = dict(seed=seed, experiment_name=spec["experiment_name"],
                       training_config_sha256=canonical_sha256(config),
                       files={str(p.relative_to(worker_dir)): sha256_file(p) for p in files})
        write_json(worker_dir / "SUCCESS.json", success)
        publish(worker_dir, remote_uri, success=True)
        return worker_dir
    except BaseException as exc:
        write_json(worker_dir / "failure.json", dict(error=str(exc), traceback=traceback.format_exc(),
                   peak_rss_mib=peak_rss_mib(), nodes_touched=getattr(solver, "nodes_touched", 0)))
        try:
            publish(worker_dir, remote_uri)
        except Exception:
            pass  # Preserve the original failure; cloud logs also retain it.
        raise
