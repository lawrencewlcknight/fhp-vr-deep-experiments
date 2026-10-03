"""VR-DeepPDCFR+ with the exact UCV Exp.2 encoder and unchanged flat MLPs.

Only input construction/dimensions and the necessary replay layout differ.
All update rules, network fitting, sampling and target-cache code are inherited.
"""

from fhp_vr_deep.features import FHPFeatureEncoder
from .solver import AvePolicyTrainer, CircularBuffer
from .variants import VRDeepPDCFRPlus, VRPDCFRPlusRegretTrainer, VRPDCFRPlusQValueTrainer


class _PlayerInput:
    def __init__(self, *args, feature_encoder, **kwargs):
        self.feature_encoder = feature_encoder
        super().__init__(*args, **kwargs)

    def get_infostate_tensor(self, state):
        # Canonicalise using this player's cards and the public board ONLY.
        return self.feature_encoder.information_state(state)


class EncodedRegretTrainer(_PlayerInput, VRPDCFRPlusRegretTrainer):
    pass


class EncodedAveragePolicyTrainer(_PlayerInput, AvePolicyTrainer):
    pass


class EncodedCriticTrainer(VRPDCFRPlusQValueTrainer):
    def __init__(self, *args, feature_encoder, **kwargs):
        self.feature_encoder = feature_encoder
        super().__init__(*args, **kwargs)

    def init_buffer(self):
        # 263 full-state features are NOT two 183-dimensional player tensors.
        # Store separately encoded next-player information, never a slice of
        # the opponent-aware, globally canonicalised critic representation.
        return CircularBuffer(self.buffer_size, self.input_size, self.state_size,
                              self.output_size, device=self.device, derive_next_state=False)

    def get_history_tensor(self, state):
        return self.feature_encoder.full_state(state)


class EncodedVRDeepPDCFRPlus(VRDeepPDCFRPlus):
    def __init__(self, *args, **kwargs):
        self.feature_encoder = FHPFeatureEncoder()
        super().__init__(*args, **kwargs)

    def information_state_size(self):
        return self.feature_encoder.policy_size

    def get_infostate_tensor(self, state):
        return self.feature_encoder.information_state(state)

    def get_history_tensor(self, state):
        return self.feature_encoder.full_state(state)

    def next_information_state(self, state, next_history, next_player):
        return self.feature_encoder.information_state(state, next_player)

    def init_ave_policy_trainer(self):
        self.ave_policy_trainer = EncodedAveragePolicyTrainer(
            self.infostate_size, self.action_size, self.network_layers,
            self.learning_rate, self.ave_policy_buffer_size, self.ave_policy_batch_size,
            self.ave_policy_network_train_steps, self.logger, self.device, self.gamma,
            feature_encoder=self.feature_encoder,
        )

    def init_regret_trainers(self):
        self.regret_trainers = [EncodedRegretTrainer(
            self.infostate_size, self.action_size, self.network_layers,
            self.learning_rate, self.advantage_buffer_size, self.advantage_batch_size,
            self.advantage_network_train_steps, self.logger,
            self.reinitialize_imm_regret_networks, self.use_regret_matching_argmax,
            self.device, self.alpha, feature_encoder=self.feature_encoder,
        ) for _ in range(self.num_players)]

    def init_q_value_trainer(self):
        self.q_value_trainer = EncodedCriticTrainer(
            self.feature_encoder.full_state_size, self.infostate_size, self.action_size,
            self.network_layers, self.learning_rate, self.baseline_buffer_size,
            self.baseline_batch_size, self.baseline_network_train_steps, self.logger,
            self.regret_trainers, self.device, feature_encoder=self.feature_encoder,
        )
