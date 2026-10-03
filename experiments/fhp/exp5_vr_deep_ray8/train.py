"""Keep the established time/checkpoint runner and substitute collection only."""

from pathlib import Path

from experiments.fhp.exp1_vr_deep_pdcfr_24h import train as baseline
from vr_deep_cfr.parallel_solver import make_solver
from . import config


def run_worker(output_root, task_index, *, smoke=False, remote_uri=None, threads=config.THREADS):
    created = []

    def factory(seed, training_config):
        solver = make_solver(seed, training_config, smoke=smoke)
        created.append(solver)
        baseline.write_json(Path(output_root) / "workers" / config.task_name(task_index) / "parallel_runtime.json",
                            solver.parallel_runtime)
        return solver

    try:
        return baseline.run_worker(output_root, task_index, smoke=smoke, remote_uri=remote_uri, threads=threads,
                                   experiment=config, solver_factory=factory)
    finally:
        # Also handles a failure between construction and solver.solve().
        for solver in created:
            solver.close()
