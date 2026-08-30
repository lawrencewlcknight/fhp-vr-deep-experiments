"""Sampled, reproducible, seat-swapped FHP policy evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fhp_vr_deep.game import FHP_GAME_PARAMETERS, load_fhp_game
from fhp_vr_deep.io_utils import sha256_file, stats
from vr_deep_cfr.policy_snapshots import LoadedVRPolicy, load_policy_snapshot_payload


BIG_BLIND_CHIPS = 100.0
MILLI_BIG_BLINDS_PER_CHIP = 1000.0 / BIG_BLIND_CHIPS


def _sample_index(probabilities, rng: np.random.Generator) -> int:
    values = np.asarray(probabilities, dtype=float)
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot sample from invalid probabilities")
    values /= total
    return int(rng.choice(len(values), p=values))


def _play_game(game, policies, chance_rng, action_rng) -> tuple[float, float]:
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            index = _sample_index([probability for _, probability in outcomes], chance_rng)
            state.apply_action(outcomes[index][0])
            continue
        player = int(state.current_player())
        probabilities = policies[player].action_probabilities(state, player)
        actions = sorted(probabilities)
        index = _sample_index([probabilities[action] for action in actions], action_rng)
        state.apply_action(actions[index])
    returns = state.returns()
    return float(returns[0]), float(returns[1])


def evaluate_snapshot_pair(
    dcfr_snapshot: str | Path,
    pdcfr_snapshot: str | Path,
    *,
    training_seed: int,
    checkpoint_target_seconds: float,
    num_deals: int,
    evaluation_seed: int,
) -> dict[str, object]:
    """Estimate DCFR+ minus PDCFR+ payoff using both seat assignments per deal."""
    if num_deals <= 0:
        raise ValueError("num_deals must be positive")
    dcfr_snapshot = Path(dcfr_snapshot)
    pdcfr_snapshot = Path(pdcfr_snapshot)
    dcfr_payload = load_policy_snapshot_payload(dcfr_snapshot)
    pdcfr_payload = load_policy_snapshot_payload(pdcfr_snapshot)
    if dcfr_payload["algorithm_id"] != "vr_deep_dcfr_plus":
        raise ValueError("The first snapshot must be VR-DeepDCFR+")
    if pdcfr_payload["algorithm_id"] != "vr_deep_pdcfr_plus":
        raise ValueError("The second snapshot must be VR-DeepPDCFR+")
    if int(dcfr_payload["seed"]) != int(training_seed):
        raise ValueError("VR-DeepDCFR+ snapshot seed does not match the evaluation pair")
    if int(pdcfr_payload["seed"]) != int(training_seed):
        raise ValueError("VR-DeepPDCFR+ snapshot seed does not match the evaluation pair")
    if dcfr_payload["game"]["parameters"] != dict(FHP_GAME_PARAMETERS):
        raise ValueError("VR-DeepDCFR+ snapshot has the wrong FHP contract")
    if pdcfr_payload["game"]["parameters"] != dict(FHP_GAME_PARAMETERS):
        raise ValueError("VR-DeepPDCFR+ snapshot has the wrong FHP contract")
    for label, payload in (("DCFR+", dcfr_payload), ("PDCFR+", pdcfr_payload)):
        stored_target = payload.get("checkpoint", {}).get("checkpoint_target_seconds")
        if stored_target is None or not np.isclose(
            float(stored_target), float(checkpoint_target_seconds)
        ):
            raise ValueError(f"{label} snapshot does not match the requested checkpoint")

    game = load_fhp_game()
    dcfr_policy = LoadedVRPolicy(game, dcfr_snapshot)
    pdcfr_policy = LoadedVRPolicy(game, pdcfr_snapshot)
    master_rng = np.random.default_rng(int(evaluation_seed))
    dcfr_player_zero = np.empty(num_deals, dtype=float)
    dcfr_player_one = np.empty(num_deals, dtype=float)

    for deal_index in range(num_deals):
        chance_seed = int(master_rng.integers(0, 2**63 - 1))
        action_seed = int(master_rng.integers(0, 2**63 - 1))
        return_zero, _ = _play_game(
            game,
            (dcfr_policy, pdcfr_policy),
            np.random.default_rng(chance_seed),
            np.random.default_rng(action_seed),
        )
        _, return_one = _play_game(
            game,
            (pdcfr_policy, dcfr_policy),
            np.random.default_rng(chance_seed),
            np.random.default_rng(action_seed),
        )
        dcfr_player_zero[deal_index] = return_zero
        dcfr_player_one[deal_index] = return_one

    paired = 0.5 * (dcfr_player_zero + dcfr_player_one)
    paired_stats = stats(paired)
    player_zero_stats = stats(dcfr_player_zero)
    player_one_stats = stats(dcfr_player_one)
    return {
        "training_seed": int(training_seed),
        "checkpoint_target_seconds": float(checkpoint_target_seconds),
        "checkpoint_target_hours": float(checkpoint_target_seconds) / 3600.0,
        "evaluation_seed": int(evaluation_seed),
        "num_deal_pairs": int(num_deals),
        "num_games": int(2 * num_deals),
        "metric": "vr_deep_dcfr_plus_minus_vr_deep_pdcfr_plus",
        "dcfr_player_zero_mean_chips": player_zero_stats["mean"],
        "dcfr_player_one_mean_chips": player_one_stats["mean"],
        "dcfr_seat_averaged_mean_chips": paired_stats["mean"],
        "dcfr_seat_averaged_std_chips": paired_stats["std"],
        "dcfr_seat_averaged_se_chips": paired_stats["se"],
        "dcfr_seat_averaged_ci95_low_chips": paired_stats["ci95_low"],
        "dcfr_seat_averaged_ci95_high_chips": paired_stats["ci95_high"],
        "dcfr_seat_averaged_mean_mbb_per_hand": (
            float(paired_stats["mean"]) * MILLI_BIG_BLINDS_PER_CHIP
        ),
        "dcfr_seat_averaged_se_mbb_per_hand": (
            float(paired_stats["se"]) * MILLI_BIG_BLINDS_PER_CHIP
        ),
        "dcfr_snapshot": str(dcfr_snapshot),
        "dcfr_snapshot_sha256": sha256_file(dcfr_snapshot),
        "pdcfr_snapshot": str(pdcfr_snapshot),
        "pdcfr_snapshot_sha256": sha256_file(pdcfr_snapshot),
        "common_random_numbers": True,
    }
