"""Same learner as Experiment 5, with a cumulative clock and final archive."""

from pathlib import Path

from experiments.fhp.exp1_vr_deep_pdcfr_24h import train as baseline
from fhp_vr_deep.io_utils import read_json, sha256_file
from vr_deep_cfr.parallel_solver import (ParallelVRDeepPDCFRPlus, RayTraversalPool,
                                         OBJECT_STORE_BYTES, SMOKE_OBJECT_STORE_BYTES)
from . import config
from .training_state import restore_state, save_state


class ResumableParallelSolver(ParallelVRDeepPDCFRPlus):
    def __init__(self, **kwargs):
        self.elapsed_before_resume = 0.
        self.safe_to_snapshot = False
        super().__init__(**kwargs)

    def _training_elapsed_seconds(self):
        return self.elapsed_before_resume + super()._training_elapsed_seconds()

    def iteration(self):
        self.safe_to_snapshot = False
        super().iteration()
        self.safe_to_snapshot = True


def make_solver(seed, training_config, *, smoke=False, pool_factory=RayTraversalPool):
    def construct(**kwargs):
        return ResumableParallelSolver(collector_config=training_config, run_seed=seed,
                                       pool_factory=pool_factory,
                                       object_store_bytes=SMOKE_OBJECT_STORE_BYTES if smoke else OBJECT_STORE_BYTES,
                                       **kwargs)
    solver = baseline.make_solver(seed, training_config, solver_class=construct)
    solver.max_wall_clock_seconds = config.TRAIN_MAX_SECONDS
    return solver


def run_worker(output_root, task_index, *, smoke=False, remote_uri=None, threads=config.THREADS,
               start_hours=0, target_hours=48, resume_run_id=None, resume_from=None):
    spec = config.RunSpec(start_hours, target_hours, resume_run_id)
    if bool(start_hours) != bool(resume_from):
        raise ValueError("Continuation requires --resume-from; fresh runs cannot load a state")
    if start_hours and smoke:
        raise ValueError("Use the restart-equivalence smoke test, not a production continuation in smoke mode")
    parent = None
    if resume_from:
        meta = read_json(Path(resume_from) / "manifest.json")
        if (meta["completed_target_hours"] != start_hours or meta["seed"] != config.SEEDS[task_index]
                or meta["is_smoke"] or meta["active_seconds"] >= spec.schedule()[0]["checkpoint_target_seconds"]):
            raise ValueError("Wrong source checkpoint/seed or source already crosses the next checkpoint")
        parent = dict(run_id=resume_run_id, manifest_sha256=sha256_file(Path(resume_from) / "manifest.json"),
                      completed_target_hours=start_hours, actual_active_seconds=meta["active_seconds"])
    created = []

    def factory(seed, training_config):
        solver = make_solver(seed, training_config, smoke=smoke)
        created.append(solver)
        if resume_from:
            restore_state(resume_from, solver, training_config=training_config, seed=seed,
                          start_hours=start_hours)
        worker = Path(output_root) / "workers" / config.task_name(task_index)
        baseline.write_json(worker / "parallel_runtime.json", solver.parallel_runtime)
        if parent:
            baseline.write_json(worker / "continuation_source.json", parent)
        return solver

    def final_state(solver, worker, final, manifest):
        directory = worker / "training_state"
        save_state(directory, solver, training_config=manifest["training_config"], seed=manifest["seed"],
                   target_hours=target_hours, elapsed_seconds=final["training_elapsed_seconds"],
                   is_smoke=smoke, parent=parent)
        return dict(final_training_state="training_state/manifest.json",
                    final_training_state_sha256=sha256_file(directory / "manifest.json"),
                    final_training_state_bytes=sum(p.stat().st_size for p in directory.iterdir()),
                    continuation_source=parent, final_training_state_verified=True)

    try:
        return baseline.run_worker(output_root, task_index, smoke=smoke, remote_uri=remote_uri,
                                   threads=threads, experiment=spec, solver_factory=factory,
                                   after_training=final_state)
    finally:
        for solver in created:
            solver.close()
