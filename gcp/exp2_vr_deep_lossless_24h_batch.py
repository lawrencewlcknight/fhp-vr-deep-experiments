#!/usr/bin/env python3
"""Encoder-only Experiment 2, using the identical Experiment 1 Batch pipeline."""

from exp1_vr_deep_pdcfr_24h_batch import main

EXPERIMENT = dict(number=2, module="experiments.fhp.exp2_vr_deep_lossless_24h",
                  algorithm_id="lossless_vr_deep_pdcfr_plus",
                  batch_script="gcp/exp2_vr_deep_lossless_24h_batch.py",
                  test_files=("tests/test_exp2_vr_deep_lossless_24h.py",
                              "tests/test_exp1_vr_deep_pdcfr_24h.py", "tests/test_efficiency.py"))


if __name__ == "__main__":
    main(experiment=EXPERIMENT)
