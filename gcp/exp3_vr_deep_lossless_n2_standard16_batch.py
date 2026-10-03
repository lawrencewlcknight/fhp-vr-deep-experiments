#!/usr/bin/env python3
"""Experiment 2's workflow, with larger smoke/training VMs only."""

from exp1_vr_deep_pdcfr_24h_batch import main

EXPERIMENT = dict(
    number=3, module="experiments.fhp.exp3_vr_deep_lossless_n2_standard16",
    algorithm_id="lossless_vr_deep_pdcfr_plus_n2_16",
    batch_script="gcp/exp3_vr_deep_lossless_n2_standard16_batch.py",
    worker_resources=dict(machine_type="n2-standard-16", cpu_milli=16000, memory_mib=62000),
    test_files=("tests/test_exp3_vr_deep_lossless_n2_standard16.py",
                "tests/test_exp2_vr_deep_lossless_24h.py",
                "tests/test_exp1_vr_deep_pdcfr_24h.py", "tests/test_efficiency.py"),
)


if __name__ == "__main__":
    main(experiment=EXPERIMENT)
