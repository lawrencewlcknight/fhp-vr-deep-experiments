import pytest

from vr_deep_cfr import VRDeepDCFRPlus, VRDeepPDCFRPlus
from vr_deep_cfr.logger import Logger


@pytest.mark.parametrize(
    ("solver_class", "variant_kwargs"),
    [
        (VRDeepDCFRPlus, {"alpha": 2.0, "gamma": 2.0}),
        (
            VRDeepPDCFRPlus,
            {
                "alpha": 2.3,
                "gamma": 2.0,
                "reinitialize_imm_regret_networks": True,
            },
        ),
    ],
)
def test_vr_deep_variant_completes_tiny_fhp_run(solver_class, variant_kwargs):
    solver = solver_class(
        game_name="FHP",
        num_episodes=8,
        advantage_buffer_size=128,
        ave_policy_buffer_size=128,
        baseline_buffer_size=128,
        learning_rate=1e-3,
        num_traversals=4,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        baseline_network_train_steps=1,
        advantage_batch_size=2,
        ave_policy_batch_size=2,
        baseline_batch_size=2,
        num_layers=1,
        num_hiddens=8,
        evaluation_frequency=1,
        reinitialize_advantage_networks=False,
        use_baseline=True,
        device="cpu",
        seed=0,
        logger=Logger(verbose=False),
        **variant_kwargs,
    )

    rows = solver.solve()

    assert len(rows) == 1
    assert rows[0]["evaluation_status"] == "deferred_to_head_to_head"
    assert rows[0]["nodes_touched"] > 0
