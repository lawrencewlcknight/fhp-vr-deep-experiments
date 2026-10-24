#!/usr/bin/env python3
"""Cloud smoke, three fresh 48-active-hour runs, verified aggregation."""
from exp1_vr_deep_pdcfr_24h_batch import main
EXPERIMENT = dict(number=8, module="experiments.fhp.exp8_vr_deep_rich_ae_48h",
    algorithm_id="rich_ae_vr_deep_pdcfr_plus_48h", batch_script="gcp/exp8_vr_deep_rich_ae_48h_batch.py",
    label="fhp-vr-exp8-rich48", worker_resources=dict(machine_type="n2-standard-16", cpu_milli=16000, memory_mib=62000),
    stage_seconds=dict(train=72*3600, controller=96*3600), extra_requirements="requirements-ray.txt",
    smoke_test_environment=dict(FHP_RUN_RAY_TESTS="1"),
    test_files=("tests/test_exp8_rich_ae.py", "tests/test_exp6_vr_deep_ray8_48h.py", "tests/test_efficiency.py"))
if __name__ == "__main__":
    main(experiment=EXPERIMENT)
