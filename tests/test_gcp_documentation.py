from pathlib import Path


def test_gcp_guide_tracks_the_fhp_vr_deep_batch_contract():
    root = Path(__file__).parents[1]
    guide = (root / "docs" / "GCP_BATCH_EXPERIMENTS.md").read_text(encoding="utf-8")

    required = (
        "fhp-vr-deep-experiments.git",
        "gcp/submit_batch_experiment.sh",
        "gcp/read_batch_task_logs.sh",
        "experiments.fhp.exp1_leduc_config_transfer.run",
        "--smoke",
        "vr_deep_dcfr_plus",
        "vr_deep_pdcfr_plus",
        "--aggregate-run-dir",
        "n2-standard-8",
        "432000",
        "pd-balanced",
        "batch_diagnostics.json",
    )
    for value in required:
        assert value in guide

    assert "leduc-poker-deep-cfr-experiments.git" not in guide

    submission_helper = (root / "gcp" / "submit_batch_experiment.sh").read_text(
        encoding="utf-8"
    )
    assert "fhp-vr-deep-experiments.git" in submission_helper
    assert "fhp-poker-vr-deep-experiments" not in submission_helper
