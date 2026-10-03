"""Encoder-only transfer: information boundaries, fitting, artifacts and cloud."""

import copy
import hashlib
import importlib.util
import itertools
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.vr_deep_efficiency import compare, run_fixed_work
from experiments.fhp.exp1_vr_deep_pdcfr_24h import config as baseline
from experiments.fhp.exp2_vr_deep_lossless_24h import config
from experiments.fhp.exp2_vr_deep_lossless_24h.aggregate import aggregate, verify_worker
from experiments.fhp.exp2_vr_deep_lossless_24h.run import main
from experiments.fhp.exp2_vr_deep_lossless_24h.train import make_solver, run_worker
from fhp_vr_deep.features import FHPFeatureEncoder, _betting_features, _betting_sequence
from fhp_vr_deep.game import load_fhp_game
from fhp_vr_deep.io_utils import read_json
from vr_deep_cfr.encoded_solver import EncodedVRDeepPDCFRPlus
from vr_deep_cfr.policy_snapshots import LoadedVRPolicy, load_policy_snapshot_payload

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def replay(history, permutation=None):
    state = load_fhp_game().new_initial_state()
    for action in history:
        if permutation is not None and state.is_chance_node():
            action = (action // 4) * 4 + permutation[action % 4]
        state.apply_action(action)
    return state


def test_exact_baseline_configuration_except_input_representation():
    assert config.TRAINING_CONFIG == baseline.TRAINING_CONFIG
    assert config.SEEDS == (0, 1, 2)
    assert config.CHECKPOINT_SECONDS == (21600, 43200, 64800, 86400)
    assert config.REFERENCE_VM == baseline.REFERENCE_VM
    assert config.contract()["network_architecture"] == "unchanged_flat_mlp_64_64_64"
    assert config.contract()["replay_storage"]["derive_critic_next_state"] is False
    solver = make_solver(0, config.smoke_config())
    assert solver.infostate_size == 183 and solver.network_layers == [64] * 3
    assert solver.q_value_trainer.input_size == 263
    assert solver.q_value_trainer.buffer.next_state_buf.shape == (128, 183)
    assert not solver.q_value_trainer.buffer.derive_next_state


def test_reference_vectors_match_pinned_ucv_encoder_exactly():
    golden = read_json(ROOT / "tests/fixtures/ucv_exp2_encoder_golden.json")
    encoder = FHPFeatureEncoder()
    assert golden["encoder_metadata"] == encoder.metadata()
    assert golden["source_commit"] == config.ENCODER_SOURCE["commit"]
    assert golden["source_file_sha256"] == config.ENCODER_SOURCE["file_sha256"]
    assert len(golden["cases"]) == 222
    for case in golden["cases"]:
        state = replay(case["history"])
        for kind, features in (("policy", encoder.information_state(state, case["player"])),
                               ("critic", encoder.full_state(state))):
            assert features.dtype == np.float32
            assert hashlib.sha256(features.tobytes()).hexdigest() == case[kind + "_sha256"]


@pytest.mark.parametrize("history", [[0, 5, 10, 15], [0, 5, 10, 15, 1, 1, 20, 25, 30]])
def test_all_24_suit_permutations_leave_features_unchanged(history):
    encoder = FHPFeatureEncoder()
    state = replay(history)
    for permutation in itertools.permutations(range(4)):
        transformed = replay(history, permutation)
        for player in (0, 1):
            np.testing.assert_array_equal(encoder.information_state(state, player),
                                          encoder.information_state(transformed, player))
        np.testing.assert_array_equal(encoder.full_state(state), encoder.full_state(transformed))


@pytest.mark.parametrize("player", [0, 1])
def test_opponent_cards_cannot_leak_into_player_inputs(player):
    # OpenSpiel deals two cards to player 0, then two to player 1.
    history = [0, 5, 10, 15]
    changed = history.copy()
    changed[2 * (1 - player):2 * (1 - player) + 2] = [36, 41]
    solver = make_solver(0, config.smoke_config())
    encoder = solver.feature_encoder
    with torch.no_grad():
        for trainer in [solver.ave_policy_trainer, *solver.regret_trainers]:
            for parameter in trainer.model.parameters():
                parameter.uniform_(-0.1, 0.1)
    for suffix in ([], [1, 1, 20, 25, 30]):
        first, second = replay(history + suffix), replay(changed + suffix)
        np.testing.assert_array_equal(first.information_state_tensor(player),
                                      second.information_state_tensor(player))
        info = encoder.information_state(first, player)
        np.testing.assert_array_equal(info, encoder.information_state(second, player))
        assert not np.array_equal(encoder.full_state(first), encoder.full_state(second))
        for trainer in [solver.ave_policy_trainer, solver.regret_trainers[player]]:
            with torch.no_grad():
                assert torch.equal(trainer.model(torch.from_numpy(info)),
                                   trainer.model(torch.from_numpy(encoder.information_state(second, player))))


def test_betting_history_is_exact_at_every_reachable_decision_history():
    encoder = FHPFeatureEncoder()
    seen = set()
    def visit(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            visit(state.child(state.chance_outcomes()[0][0]))
            return
        for player in (0, 1):
            expected = np.concatenate(_betting_features(_betting_sequence(state, player)))
            np.testing.assert_array_equal(encoder.information_state(state, player)[106:138], expected)
        seen.add(_betting_sequence(state, 0))
        for action in state.legal_actions():
            visit(state.child(action))
    visit(load_fhp_game().new_initial_state())
    assert len(seen) > 50


def test_next_player_input_is_encoded_separately_not_sliced_from_critic(monkeypatch):
    solver = make_solver(0, config.smoke_config())
    solver.num_iteration = 1  # Collection normally starts inside iteration one.
    actual = solver.next_information_state
    recorded = []
    def capture(state, history, player):
        result = actual(state, history, player)
        np.testing.assert_array_equal(result, solver.feature_encoder.information_state(state, player))
        assert result.shape == (183,) and history.shape == (263,)
        recorded.append(result.copy())
        return result
    monkeypatch.setattr(solver, "next_information_state", capture)
    solver.collect_training_data(0)
    assert recorded
    buffer = solver.q_value_trainer.buffer
    nonterminal = buffer.done_buf[:len(buffer)].reshape(-1) == 0
    for row in buffer.next_state_buf[:len(buffer)][nonterminal]:
        assert any(np.array_equal(row, expected) for expected in recorded)
    assert np.all(buffer.next_state_buf[:len(buffer)][~nonterminal] == 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_encoded_cached_and_uncached_training_are_identical(seed):
    module = SimpleNamespace(VRDeepPDCFRPlus=EncodedVRDeepPDCFRPlus)
    settings = dict(seed=seed, iterations=3, traversals=32, capacity=71, batch=16,
                    regret_steps=9, critic_steps=101, policy_steps=9, layers=2, width=16)
    compare(run_fixed_work(module, cache=False, **settings), run_fixed_work(module, **settings))


def test_encoded_production_batch_cache_is_exact_with_eight_threads():
    from vr_deep_cfr.solver import MLP
    from vr_deep_cfr.frozen_cache import build_frozen_cache
    torch.set_num_threads(8)
    torch.manual_seed(81)
    model = MLP(183, [64] * 3, 3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    inputs = torch.randn(4171, 183)
    cache = build_frozen_cache(4171, 2048, 750, lambda i: model(inputs[i]), device="cpu")
    assert cache is not None
    for _ in range(3):
        indices = torch.randperm(4171)[:2048].numpy()
        with torch.no_grad():
            assert torch.equal(torch.from_numpy(cache[indices]), model(inputs[indices]))


def test_encoded_clock_finishes_both_players_before_checkpoint(monkeypatch):
    solver = make_solver(0, config.smoke_config())
    solver.training_time_checkpoint_seconds = (1, 2, 3, 4)
    calls, elapsed = [], [0.0]
    monkeypatch.setattr(solver, "_training_elapsed_seconds", lambda: elapsed[0])
    def collect(player):
        calls.append(("collect", player))
        elapsed[0] += 3
        solver._maybe_run_training_time_checkpoint()
        assert not solver._stop_requested
        assert not any(c[0] == "checkpoint" for c in calls)
    monkeypatch.setattr(solver, "collect_training_data", collect)
    monkeypatch.setattr(solver, "train_regret", lambda p: calls.append(("regret", p)))
    monkeypatch.setattr(solver, "train_baseline", lambda p: calls.append(("critic", p)))
    solver.ave_policy_trainer.buffer.add(np.zeros(183), [1, 0, 0], [1, 1, 1], 1)
    monkeypatch.setattr(solver, "_run_checkpoint", lambda **kw: calls.append(("checkpoint", kw["checkpoint_target_seconds"])))
    solver.solve()
    assert calls == [("collect", 0), ("regret", 0), ("critic", 0),
                     ("collect", 1), ("regret", 1), ("critic", 1)] + [
                         ("checkpoint", x) for x in (1, 2, 3, 4)]
    assert solver.num_iteration == 1 and solver.stop_reason == "training_time_budget"


@pytest.fixture(scope="module")
def smoke_output(tmp_path_factory):
    root = tmp_path_factory.mktemp("vr-exp2")
    previous = torch.get_num_threads()
    try:
        assert main(["smoke", "--output-root", str(root), "--threads", "1"]) == 0
    finally:
        torch.set_num_threads(previous)
    return root


def test_playable_snapshots_and_relocatable_aggregation(smoke_output, tmp_path):
    root = tmp_path / "download"
    shutil.copytree(smoke_output, root)
    result = aggregate(root, smoke=True)
    worker = root / "workers" / config.task_name(0)
    manifest, _, snapshots = verify_worker(worker, smoke=True)
    assert manifest["feature_encoder"] == FHPFeatureEncoder().metadata()
    assert read_json(result / "summary.json")["policy_count"] == 4
    assert len(list(worker.rglob("*.pt"))) == 4
    for row in snapshots:
        payload = load_policy_snapshot_payload(worker / row["path"])
        assert payload["version"] == 3 and payload["input_size"] == 183
        assert payload["policy_network_layers"] == [64] * 3
        assert not {"optimizer", "replay", "training_state"} & payload.keys()
        target = LoadedVRPolicy(load_fhp_game(), worker / row["path"])
        for history in ([0, 5, 10, 15], [0, 5, 10, 15, 1, 1, 20, 25, 30]):
            state = replay(history)
            probs = target.action_probabilities(state)
            assert np.isclose(sum(probs.values()), 1.0)
            assert set(probs) == set(state.legal_actions())
            with torch.no_grad():
                logits = target.model(torch.from_numpy(FHPFeatureEncoder().information_state(state)))
                expected = torch.softmax(logits[state.legal_actions()], -1).numpy()
            np.testing.assert_allclose(list(probs.values()), expected, rtol=1e-6)
    from experiments.fhp.exp1_vr_deep_pdcfr_24h.aggregate import verify_worker as verify_raw
    with pytest.raises(ValueError):
        verify_raw(worker, smoke=True)


def test_bad_encoder_snapshot_metadata_fails_closed(smoke_output, tmp_path):
    payload = load_policy_snapshot_payload(next((smoke_output / "workers").rglob("*.pt")))
    for key, value in (("feature_encoder", None), ("feature_encoder", {"id": "wrong"}),
                       ("input_size", 190), ("version", 2)):
        bad = copy.deepcopy(payload)
        bad[key] = value
        path = tmp_path / "bad.pt"
        torch.save(bad, path)
        with pytest.raises(ValueError):
            load_policy_snapshot_payload(path)


def test_shared_evaluation_adapter_supports_encoded_policies(smoke_output):
    from fhp_vr_deep.evaluation_adapter import _import_suite, load_policy_for_evaluation
    try:
        _import_suite()
    except ModuleNotFoundError:
        pytest.skip("Shared suite is optional for training")
    path = next((smoke_output / "workers").rglob("*.pt"))
    game, target = load_policy_for_evaluation(path)
    expected = LoadedVRPolicy(game, path)
    for history in ([0, 5, 10, 15], [0, 5, 10, 15, 1, 1, 20, 25, 30]):
        assert target.action_probabilities(replay(history)) == expected.action_probabilities(replay(history))


def test_three_seed_path_shortened_only_inside_test(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAINING_CONFIG", config.smoke_config())
    monkeypatch.setattr(config, "CHECKPOINT_SECONDS", config.SMOKE_SECONDS)
    for index in range(3):
        run_worker(tmp_path, index)
    summary = read_json(aggregate(tmp_path) / "summary.json")
    assert summary["seeds"] == [0, 1, 2] and summary["policy_count"] == 12


@pytest.mark.parametrize("stage", ["controller", "smoke", "train", "aggregate"])
def test_cloud_resources_and_generated_shell(stage):
    spec = importlib.util.spec_from_file_location("vr_base_batch", ROOT / "gcp/exp1_vr_deep_pdcfr_24h_batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    args = SimpleNamespace(project="test-project", region="europe-west1", bucket="gs://test-bucket",
                           service_account="worker@test.iam.gserviceaccount.com", repo_ref="a" * 40,
                           run_id="vr2-encode-test", experiment=dict(number=2,
                               module="experiments.fhp.exp2_vr_deep_lossless_24h",
                               algorithm_id=config.ALGORITHM_ID,
                               batch_script="gcp/exp2_vr_deep_lossless_24h_batch.py",
                               test_files=("tests/test_exp2_vr_deep_lossless_24h.py",)))
    job = batch.build_job(args, stage)
    text = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    assert "exp2_vr_deep_lossless_24h" in text and "fhp-ucv-escher" not in text
    assert job["labels"]["experiment"] == "fhp-vr-exp2-24h"
    if stage == "train":
        group = job["taskGroups"][0]
        assert group["taskCount"] == group["parallelism"] == 3 and group["taskCountPerNode"] == 1
        assert group["taskSpec"]["maxRunDuration"] == "129600s"
        assert group["taskSpec"]["maxRetryCount"] == 0
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    if stage == "smoke":
        assert "test_exp2_vr_deep_lossless_24h.py" in text and "benchmarks.vr_deep_efficiency" in text
