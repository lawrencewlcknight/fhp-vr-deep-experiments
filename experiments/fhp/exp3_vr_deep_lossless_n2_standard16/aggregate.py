"""Training diagnostics only; no poker-performance evaluation."""

from experiments.fhp.exp1_vr_deep_pdcfr_24h import aggregate as baseline
from . import config


def verify_worker(worker_dir, *, smoke=False):
    return baseline.verify_worker(worker_dir, smoke=smoke, experiment=config)


def aggregate(output_root, *, smoke=False):
    return baseline.aggregate(output_root, smoke=smoke, experiment=config)
