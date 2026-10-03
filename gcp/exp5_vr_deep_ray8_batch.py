#!/usr/bin/env python3
"""Same Experiment 3 VMs/time budget, eight traversal actors per seed VM."""

from exp1_vr_deep_pdcfr_24h_batch import main

EXPERIMENT = dict(
    number=5, module="experiments.fhp.exp5_vr_deep_ray8",
    algorithm_id="lossless_vr_deep_pdcfr_plus_ray8",
    batch_script="gcp/exp5_vr_deep_ray8_batch.py",
    worker_resources=dict(machine_type="n2-standard-16", cpu_milli=16000, memory_mib=62000),
    extra_requirements="requirements-ray.txt",
    smoke_test_environment=dict(FHP_RUN_RAY_TESTS="1"),
    test_files=("tests/test_exp5_vr_deep_ray8.py", "tests/test_vr_deep_parallel.py",
                "tests/test_exp3_vr_deep_lossless_n2_standard16.py", "tests/test_efficiency.py"),
)


if __name__ == "__main__":
    main(experiment=EXPERIMENT)
