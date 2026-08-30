"""Lightweight, playable average-policy snapshots for the VR-Deep solvers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from open_spiel.python import policy

from fhp_vr_deep.game import FHP_GAME_PARAMETERS, serialisable_game_definition

from .solver import MLP


SNAPSHOT_TYPE = "vr_deep_cfr_policy_snapshot"
SNAPSHOT_VERSION = 2


def snapshot_filename(algorithm_id: str, seed: int) -> str:
    return f"{algorithm_id}_seed_{int(seed)}_final_policy_snapshot.pt"


def save_policy_snapshot(
    solver,
    path: str | Path,
    *,
    algorithm_id: str,
    algorithm_label: str,
    seed: int,
    config: Dict[str, Any],
    checkpoint_row: Dict[str, Any] | None = None,
) -> Path:
    """Save only the fitted final average-policy and provenance metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in solver.ave_policy_trainer.model.state_dict().items()
    }
    torch.save(
        {
            "version": SNAPSHOT_VERSION,
            "type": SNAPSHOT_TYPE,
            "algorithm_id": str(algorithm_id),
            "algorithm_label": str(algorithm_label),
            "game": serialisable_game_definition(),
            "seed": int(seed),
            "nodes_touched": int(solver.nodes_touched),
            "iteration": int(solver.num_iteration),
            "episode": int(solver.episode),
            "checkpoint": dict(checkpoint_row or {}),
            "policy_state_dict": state_dict,
            "policy_network_layers": list(solver.network_layers),
            "input_size": int(solver.infostate_size),
            "num_actions": int(solver.action_size),
            "authors_parameterisation": dict(config),
        },
        path,
    )
    return path


def load_policy_snapshot_payload(path: str | Path) -> dict:
    """Load and validate snapshot identity and the exact FHP game contract."""
    snapshot = torch.load(Path(path), map_location="cpu", weights_only=False)
    if snapshot.get("type") != SNAPSHOT_TYPE:
        raise ValueError(f"Not a VR-Deep policy snapshot: {path}")
    if int(snapshot.get("version", -1)) != SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported VR-Deep snapshot version: {snapshot.get('version')!r}")
    parameters = snapshot.get("game", {}).get("parameters")
    if parameters != dict(FHP_GAME_PARAMETERS):
        raise ValueError("Snapshot does not use the canonical FHP game definition")
    return snapshot


class LoadedVRPolicy(policy.Policy):
    """OpenSpiel policy backed by a saved VR-Deep average-policy network."""

    def __init__(self, game, snapshot_path: str | Path):
        super().__init__(game, list(range(game.num_players())))
        self.path = Path(snapshot_path)
        snapshot = load_policy_snapshot_payload(self.path)
        self.metadata = {
            key: value for key, value in snapshot.items() if key != "policy_state_dict"
        }
        self.model = MLP(
            int(snapshot["input_size"]),
            [int(value) for value in snapshot["policy_network_layers"]],
            int(snapshot["num_actions"]),
        )
        self.model.load_state_dict(snapshot["policy_state_dict"], strict=True)
        self.model.eval()

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal_actions = list(state.legal_actions(player))
        if not legal_actions:
            return {}
        info_state = torch.as_tensor(
            state.information_state_tensor(player), dtype=torch.float32
        )
        legal_mask = torch.as_tensor(
            state.legal_actions_mask(player), dtype=torch.float32
        )
        with torch.no_grad():
            logits = self.model(info_state)
            legal_logits = torch.where(
                legal_mask == 1,
                logits,
                torch.full_like(logits, -1e21),
            )
            probabilities = torch.softmax(legal_logits, dim=-1).cpu().numpy()
        selected = {action: float(probabilities[action]) for action in legal_actions}
        total = float(sum(selected.values()))
        if total <= 0 or not np.isfinite(total):
            raise ValueError(f"Invalid policy probabilities in {self.path}")
        return {action: value / total for action, value in selected.items()}


__all__ = [
    "LoadedVRPolicy",
    "SNAPSHOT_TYPE",
    "load_policy_snapshot_payload",
    "save_policy_snapshot",
    "snapshot_filename",
]
