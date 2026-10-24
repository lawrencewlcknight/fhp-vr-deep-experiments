"""Rich-input equality, leakage, learner transport and artifact contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fhp_vr_deep.rich_ae import make_encoder, encoder_from_metadata, provenance, ENCODER_ID
from fhp_vr_deep.rich_ae.game import load_fhp_game

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/rich_ae_ucv_exp38.json").read_text())

@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    yield
    torch.set_num_threads(before)

def test_matches_pinned_ucv_player_and_critic_inputs_exactly():
    enc = make_encoder()
    assert enc.metadata() == FIXTURE["metadata"]
    assert provenance()["commit"] == FIXTURE["source_commit"]
    for row in FIXTURE["states"]:
        state = load_fhp_game().new_initial_state()
        for action in row["history"]:
            state.apply_action(action)
        for player in (0, 1):
            np.testing.assert_array_equal(enc.information_state(state, player), np.asarray(row["player"][player], np.float32))
        np.testing.assert_array_equal(enc.full_state(state), np.asarray(row["critic"], np.float32))
    assert encoder_from_metadata(enc.metadata()).metadata() == enc.metadata()
    with pytest.raises(ValueError):
        encoder_from_metadata(dict(enc.metadata(), opponent_prior="actual hidden cards"))

def test_player_inputs_do_not_leak_opponent_cards_and_are_suit_invariant():
    enc = make_encoder()
    histories = [[48,45,3,7,1,1,24,30,34], [48,45,9,11,1,1,24,30,34]]
    states = []
    for history in histories:
        state = load_fhp_game().new_initial_state()
        for action in history: state.apply_action(action)
        states.append(state)
    np.testing.assert_array_equal(enc.information_state(states[0],0),enc.information_state(states[1],0))
    assert not np.array_equal(enc.full_state(states[0]),enc.full_state(states[1]))
    permutation = [2,0,3,1]
    state = load_fhp_game().new_initial_state()
    for action in histories[0]:
        state.apply_action(action//4*4+permutation[action%4] if state.is_chance_node() else action)
    for player in (0,1):
        np.testing.assert_array_equal(enc.information_state(states[0],player),enc.information_state(state,player))
    np.testing.assert_array_equal(enc.full_state(states[0]),enc.full_state(state))

from experiments.fhp.exp8_vr_deep_rich_ae_48h import config,train
from experiments.fhp.exp8_vr_deep_rich_ae_48h.aggregate import aggregate,verify_worker
from experiments.fhp.exp6_vr_deep_ray8_48h import config as baseline
from experiments.fhp.exp6_vr_deep_ray8_48h.training_state import verify_state
from vr_deep_cfr.policy_snapshots import LoadedVRPolicy
from test_vr_deep_parallel import InlinePool

def test_only_representation_changes_including_all_eight_collectors():
    for smoke in (False,True):
        cfg = config.smoke_config() if smoke else dict(config.TRAINING_CONFIG)
        assert cfg.pop("feature_encoder_id") == ENCODER_ID
        assert cfg == (baseline.smoke_config() if smoke else baseline.TRAINING_CONFIG)
    assert config.SEEDS == (0,1,2) and config.CHECKPOINT_SECONDS[-1] == 48*3600
    solver = train.make_solver(0,config.smoke_config(),smoke=True,pool_factory=InlinePool)
    try:
        assert solver.infostate_size == 231 and solver.q_value_trainer.input_size == 359
        assert all(w.solver.feature_encoder.metadata()==make_encoder().metadata() for w in solver.pool.workers)
        solver.iteration()
        critic = solver.q_value_trainer.buffer
        assert critic.next_state_buf.shape[1] == 231
        assert np.isfinite(critic.history_buf[:len(critic)]).all()
    finally: solver.close()

def test_complete_worker_reload_and_final_state(tmp_path,monkeypatch):
    original = train.make_solver
    monkeypatch.setattr(train,"make_solver",lambda seed,cfg,smoke=False:
        original(seed,cfg,smoke=smoke,pool_factory=InlinePool))
    worker = train.run_worker(tmp_path,0,smoke=True,threads=1)
    manifest,summary,rows = verify_worker(worker,smoke=True)
    assert len(rows)==8 and summary["input_sizes"]["critic"]==359
    policy = LoadedVRPolicy(load_fhp_game(),worker/rows[-1]["path"])
    state = load_fhp_game().new_initial_state()
    for action in FIXTURE["states"][1]["history"]: state.apply_action(action)
    assert sum(policy.action_probabilities(state).values()) == pytest.approx(1)
    with pytest.raises(ValueError): verify_state(worker/"training_state") # old Exp6 identity must reject it
    analysis = aggregate(tmp_path,smoke=True)
    assert json.loads((analysis/"summary.json").read_text())["policy_count"]==8
    damaged=worker/"training_state/learner.pt"
    damaged.write_bytes(damaged.read_bytes()+b"corrupt")
    with pytest.raises(ValueError): aggregate(tmp_path,smoke=True)

def test_cloud_three_seed_plan_and_shell_syntax(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"gcp"))
    import exp1_vr_deep_pdcfr_24h_batch as shared
    import exp8_vr_deep_rich_ae_48h_batch as cloud
    args=SimpleNamespace(project="test",region="europe-west1",bucket="gs://test",
        service_account="runner@test",repo_ref="a"*40,run_id="vr8-test",experiment=cloud.EXPERIMENT)
    for stage in ("controller","smoke","train","aggregate"):
        job=shared.build_job(args,stage);group=job["taskGroups"][0]
        script=group["taskSpec"]["runnables"][0]["script"]["text"]
        subprocess.run(["bash","-n"],input=script,text=True,check=True)
        assert "--resume-from" not in script
        if stage=="train":
            assert group["taskCount"] == group["parallelism"] == 3
            assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"]=="n2-standard-16"
    subprocess.run(["bash","-n",str(ROOT/"gcp/run_exp8_vr_deep_rich_ae_48h.sh")],check=True)

@pytest.mark.skipif(os.environ.get("FHP_RUN_RAY_TESTS") != "1",reason="cloud real-Ray gate")
def test_real_ray_rich_worker(tmp_path):
    train.run_worker(tmp_path,0,smoke=True,threads=1)
    aggregate(tmp_path,smoke=True)
