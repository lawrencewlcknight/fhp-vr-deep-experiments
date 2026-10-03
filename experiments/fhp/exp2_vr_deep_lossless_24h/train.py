"""Reuse Experiment 1's time protocol with only an encoded solver substitution."""

from experiments.fhp.exp1_vr_deep_pdcfr_24h import train as baseline
from vr_deep_cfr.encoded_solver import EncodedVRDeepPDCFRPlus
from . import config


class TimedEncodedVRDeepPDCFRPlus(EncodedVRDeepPDCFRPlus, baseline.TimedVRDeepPDCFRPlus):
    pass


def make_solver(seed, training_config):
    return baseline.make_solver(seed, training_config, solver_class=TimedEncodedVRDeepPDCFRPlus)


def run_worker(output_root, task_index, *, smoke=False, remote_uri=None, threads=config.THREADS):
    return baseline.run_worker(output_root, task_index, smoke=smoke, remote_uri=remote_uri,
                               threads=threads, experiment=config, solver_factory=make_solver)
