from pathlib import Path

from experiments.fhp.archieved_exp1_leduc_config_transfer.config import (
    ALGORITHMS,
    APPROVAL_STATUS,
    APPROVED_CONFIG,
    BATCH_TIMEOUT_SECONDS,
    CHECKPOINT_TRAINING_SECONDS,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    LEDUC_SOURCE_CONFIG,
    PROPOSED_CONFIG,
    PROPOSED_PROTOCOL,
    UPSTREAM,
    validate_config,
    validate_proposal,
)


def test_archived_namespace_preserves_historical_id_and_frees_active_package():
    root = Path(__file__).resolve().parents[1]
    assert EXPERIMENT_ID == 1
    assert EXPERIMENT_NAME == "archieved_exp1_fhp_vr_deep_leduc_config_transfer"
    assert (root / "experiments/fhp/archieved_exp1_leduc_config_transfer/run.py").is_file()
    assert not (root / "experiments/fhp/exp1_leduc_config_transfer/run.py").exists()


def test_only_training_config_change_is_the_game():
    validate_proposal()
    changed = {
        key
        for key in LEDUC_SOURCE_CONFIG
        if LEDUC_SOURCE_CONFIG[key] != PROPOSED_CONFIG[key]
    }
    assert changed == {"game_name"}
    assert PROPOSED_CONFIG["game_name"] == "FHP"


def test_both_vr_deep_variants_and_paired_seeds_are_present():
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert set(ALGORITHMS) == {"vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"}
    assert ALGORITHMS["vr_deep_dcfr_plus"]["alpha"] == 2.0
    assert ALGORITHMS["vr_deep_pdcfr_plus"]["alpha"] == 2.3
    assert ALGORITHMS["vr_deep_pdcfr_plus"]["reinitialize_imm_regret_networks"] is True


def test_proposed_protocol_is_time_bound_and_seat_swapped():
    assert PROPOSED_PROTOCOL["training_time_checkpoint_hours"] == [6, 12]
    assert PROPOSED_PROTOCOL["stop_after_final_training_time_checkpoint"] is True
    assert PROPOSED_PROTOCOL["sampled_seat_swapped_head_to_head"] is True


def test_upstream_commit_is_pinned_in_notice():
    notice = Path(__file__).parents[1] / "vr_deep_cfr" / "UPSTREAM.md"
    assert UPSTREAM["commit"] in notice.read_text(encoding="utf-8")


def test_approved_production_configuration_is_runnable_and_exact():
    assert APPROVAL_STATUS == "approved"
    assert CHECKPOINT_TRAINING_SECONDS == (6 * 60 * 60, 12 * 60 * 60)
    assert BATCH_TIMEOUT_SECONDS == 120 * 60 * 60
    validate_config(APPROVED_CONFIG, production=True)
