"""Reuse the exact Experiment 2 solver and Experiment 1 checkpoint protocol."""

from experiments.fhp.exp1_vr_deep_pdcfr_24h import train as baseline
from experiments.fhp.exp2_vr_deep_lossless_24h.train import make_solver
from . import config


def run_worker(output_root, task_index, *, smoke=False, remote_uri=None, threads=config.THREADS):
    return baseline.run_worker(output_root, task_index, smoke=smoke, remote_uri=remote_uri,
                               threads=threads, experiment=config, solver_factory=make_solver)
