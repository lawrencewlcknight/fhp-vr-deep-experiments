"""Approved configuration for FHP Archived Experiment 1."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Mapping


APPROVAL_STATUS = "approved"
# Preserve the historical numeric ID; the archived name is a separate namespace.
EXPERIMENT_ID = 1
EXPERIMENT_NAME = "archieved_exp1_fhp_vr_deep_leduc_config_transfer"
DEFAULT_SEEDS = [0, 1, 2]
CHECKPOINT_TRAINING_SECONDS = (6 * 60 * 60, 12 * 60 * 60)
HEAD_TO_HEAD_DEALS_PER_PAIR = 100_000
BATCH_TIMEOUT_SECONDS = 120 * 60 * 60
SPLIT_BATCH_TIMEOUT_SECONDS = 60 * 60 * 60
REFERENCE_VM = {
    "machine_type": "n2-standard-8",
    "cpu_milli": 8_000,
    "memory_mib": 32_000,
    "boot_disk_gib": 100,
    "boot_disk_type": "pd-balanced",
}

ALGORITHMS = {
    "vr_deep_dcfr_plus": {
        "algorithm_id": "vr_deep_dcfr_plus",
        "algorithm_label": "VR-DeepDCFR+",
        "class_name": "VRDeepDCFRPlus",
        "alpha": 2.0,
        "gamma": 2.0,
        "reinitialize_imm_regret_networks": None,
    },
    "vr_deep_pdcfr_plus": {
        "algorithm_id": "vr_deep_pdcfr_plus",
        "algorithm_label": "VR-DeepPDCFR+",
        "class_name": "VRDeepPDCFRPlus",
        "alpha": 2.3,
        "gamma": 2.0,
        "reinitialize_imm_regret_networks": True,
    },
}

# Verbatim algorithm/training settings used by the Leduc comparison in
# escher-architecture (Table 2 settings, rather than the differing released
# YAML values). Keep this frozen so later FHP experiments cannot silently
# redefine what "the Leduc configuration" means.
LEDUC_SOURCE_CONFIG = {
    "game_name": "leduc_poker",
    "advantage_buffer_size": 1_000_000,
    "ave_policy_buffer_size": 1_000_000,
    "baseline_buffer_size": 1_000_000,
    "learning_rate": 1e-3,
    "num_traversals": 10_000,
    "advantage_network_train_steps": 750,
    "ave_policy_network_train_steps": 5_000,
    "baseline_network_train_steps": 10_000,
    "advantage_batch_size": 2_048,
    "ave_policy_batch_size": 2_048,
    "baseline_batch_size": 2_048,
    "num_layers": 3,
    "num_hiddens": 64,
    "reinitialize_advantage_networks": False,
    "use_regret_matching_argmax": True,
    "epsilon": 0.6,
    "fit_advantage": True,
    "use_baseline": True,
    "device": "cpu",
    "evaluation_frequency": 1,
    "max_num_iterations": 100,
    "preserve_evaluation_rng": True,
}

PROPOSED_CONFIG = deepcopy(LEDUC_SOURCE_CONFIG)
PROPOSED_CONFIG["game_name"] = "FHP"
APPROVED_CONFIG = deepcopy(PROPOSED_CONFIG)

# These govern experimental observation and stopping, not the update rule.
PROPOSED_PROTOCOL = {
    "training_time_checkpoint_hours": [6, 12],
    "stop_after_final_training_time_checkpoint": True,
    "save_reloadable_policy_at_each_checkpoint": True,
    "sampled_seat_swapped_head_to_head": True,
    "head_to_head_deals_per_pair": 100_000,
    "report_mean_and_standard_error": True,
}
APPROVED_PROTOCOL = deepcopy(PROPOSED_PROTOCOL)

TRAINING_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(
        APPROVED_CONFIG,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

UPSTREAM = {
    "repository": "https://github.com/rpSebastian/DeepPDCFR",
    "commit": "9f156c9fcdac7f8c9bd0debf94c9432d222858d3",
    "leduc_source_repository": (
        "https://github.com/lawrencewlcknight/"
        "leduc-poker-escher-architecture-experiments"
    ),
    "leduc_source_config": (
        "experiments/leduc_poker/escher_vs_vr_deep_cfr_matched_nodes/config.py"
    ),
}


def validate_proposal() -> None:
    """Reject drift from the intended one-field Leduc-to-FHP transfer."""
    changed = {
        key
        for key in LEDUC_SOURCE_CONFIG
        if PROPOSED_CONFIG.get(key) != LEDUC_SOURCE_CONFIG[key]
    }
    if changed != {"game_name"}:
        raise ValueError(f"Unexpected Leduc configuration changes: {sorted(changed)}")
    if PROPOSED_CONFIG["game_name"] != "FHP":
        raise ValueError("Archived Experiment 1 must use the canonical FHP loader")
    if set(ALGORITHMS) != {"vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"}:
        raise ValueError("Archived Experiment 1 must run both VR-Deep variants")
    checkpoints = PROPOSED_PROTOCOL["training_time_checkpoint_hours"]
    if checkpoints != sorted(set(checkpoints)) or any(value <= 0 for value in checkpoints):
        raise ValueError("Checkpoint hours must be unique, positive, and increasing")


def validate_config(config: Mapping[str, object], *, production: bool) -> None:
    """Validate the game/solver contract and optionally require exact production settings."""
    validate_proposal()
    if config.get("game_name") != "FHP":
        raise ValueError("Archived Experiment 1 must use the canonical FHP loader")
    if not bool(config.get("use_baseline")):
        raise ValueError("Both approved VR-Deep variants require the history-value baseline")
    if not bool(config.get("preserve_evaluation_rng")):
        raise ValueError("Checkpoint fitting must preserve training RNG state")
    if production and dict(config) != APPROVED_CONFIG:
        raise ValueError("Production configuration has drifted from the approved transfer")


def smoke_config() -> dict:
    """Use tiny capacities and fitting counts without changing algorithm mechanisms."""
    config = deepcopy(APPROVED_CONFIG)
    config.update(
        {
            "advantage_buffer_size": 128,
            "ave_policy_buffer_size": 128,
            "baseline_buffer_size": 192,
            "num_traversals": 2,
            "advantage_network_train_steps": 1,
            "ave_policy_network_train_steps": 1,
            "baseline_network_train_steps": 1,
            "advantage_batch_size": 2,
            "ave_policy_batch_size": 2,
            "baseline_batch_size": 2,
            "num_layers": 1,
            "num_hiddens": 8,
        }
    )
    return config
