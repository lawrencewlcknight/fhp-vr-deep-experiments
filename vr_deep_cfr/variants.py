import math

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .logger import Logger
from .frozen_cache import build_frozen_cache, cached_tensor

from .solver import MLP, DeepCumuAdv, RegretTrainer, QValueTrainer, SpielState



class VRDeepDCFRPlus(DeepCumuAdv):
    def __init__(
        self,
        game_name,
        num_episodes=400,
        advantage_buffer_size=100_0000,
        ave_policy_buffer_size=100_0000,
        learning_rate=1e-4,
        num_traversals=20,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        advantage_batch_size=-1,
        ave_policy_batch_size=-1,
        num_layers=2,
        num_hiddens=128,
        evaluation_frequency=10,
        reinitialize_advantage_networks=True,
        use_regret_matching_argmax=True,
        epsilon=0.6,
        logger=None,
        fit_advantage=True,
        use_baseline=False,
        baseline_buffer_size=100_0000,
        baseline_batch_size=-1,
        baseline_network_train_steps=1,
        gamma=4,
        play_against_random=False,
        num_random_games=20000,
        alpha=1.5,
        device="cpu",
        seed=0,
    ):
        self.alpha = alpha
        super().__init__(
            game_name,
            num_episodes,
            advantage_buffer_size,
            ave_policy_buffer_size,
            learning_rate,
            num_traversals,
            advantage_network_train_steps,
            ave_policy_network_train_steps,
            advantage_batch_size,
            ave_policy_batch_size,
            num_layers,
            num_hiddens,
            evaluation_frequency,
            reinitialize_advantage_networks,
            use_regret_matching_argmax,
            epsilon,
            logger,
            fit_advantage,
            use_baseline,
            baseline_buffer_size,
            baseline_batch_size,
            baseline_network_train_steps,
            gamma,
            play_against_random,
            num_random_games,
            device,
            seed,
        )

    def init_regret_trainers(self):
        self.regret_trainers = [
            VRDCFRPlusRegretTrainer(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.use_regret_matching_argmax,
                self.device,
                self.alpha,
            )
            for _ in range(self.num_players)
        ]



class VRDeepPDCFRPlus(DeepCumuAdv):
    def __init__(
        self,
        game_name,
        num_episodes=400,
        advantage_buffer_size=100_0000,
        ave_policy_buffer_size=100_0000,
        learning_rate=1e-4,
        num_traversals=20,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        advantage_batch_size=-1,
        ave_policy_batch_size=-1,
        num_layers=2,
        num_hiddens=128,
        evaluation_frequency=10,
        reinitialize_advantage_networks=True,
        reinitialize_imm_regret_networks=True,
        use_regret_matching_argmax=True,
        epsilon=0.6,
        logger=None,
        fit_advantage=True,
        use_baseline=False,
        baseline_buffer_size=100_0000,
        baseline_batch_size=-1,
        baseline_network_train_steps=1,
        alpha=2.3,
        gamma=5,
        play_against_random=False,
        num_random_games=20000,
        device="cpu",
        seed=0,
    ):
        self.alpha = alpha
        self.reinitialize_imm_regret_networks = reinitialize_imm_regret_networks
        super().__init__(
            game_name,
            num_episodes,
            advantage_buffer_size,
            ave_policy_buffer_size,
            learning_rate,
            num_traversals,
            advantage_network_train_steps,
            ave_policy_network_train_steps,
            advantage_batch_size,
            ave_policy_batch_size,
            num_layers,
            num_hiddens,
            evaluation_frequency,
            reinitialize_advantage_networks,
            use_regret_matching_argmax,
            epsilon,
            logger,
            fit_advantage,
            use_baseline,
            baseline_buffer_size,
            baseline_batch_size,
            baseline_network_train_steps,
            gamma,
            play_against_random,
            num_random_games,
            device,
            seed,
        )

    def init_regret_trainers(self):
        self.regret_trainers = [
            VRPDCFRPlusRegretTrainer(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.reinitialize_imm_regret_networks,
                self.use_regret_matching_argmax,
                self.device,
                self.alpha,
            )
            for _ in range(self.num_players)
        ]

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_tensor = np.append(
            root_state.information_state_tensor(0),
            root_state.information_state_tensor(1),
        )
        history_size = len(history_tensor)
        self.q_value_trainer = VRPDCFRPlusQValueTrainer(
            history_size,
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.baseline_buffer_size,
            self.baseline_batch_size,
            self.baseline_network_train_steps,
            self.logger,
            self.regret_trainers,
            self.device,
        )

class VRDCFRPlusRegretTrainer(RegretTrainer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        use_regret_matching_argmax: bool,
        device: str,
        alpha: float,
    ):
        self.alpha = alpha
        super().__init__(
            input_size,
            output_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            use_regret_matching_argmax,
            device,
        )

    def compute_loss(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, iterations = samples
        with torch.no_grad():
            target_outputs = self.predict(
                self.target_model, infostates, legal_actions_mask
            )
        zero = torch.zeros_like(target_outputs)
        target_outputs = torch.maximum(target_outputs, zero)
        regrets = (
            target_outputs
            * math.pow(T - 1, self.alpha)
            / (math.pow(T - 1, self.alpha) + 1.5)
            + cf_regrets
        )
        outputs = self.predict(self.model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, regrets)
        return loss


class VRPDCFRPlusRegretTrainer(RegretTrainer):
    cache_frozen_predictions = True

    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        reinitialize_imm_regret_networks: bool,
        use_regret_matching_argmax: bool,
        device: str = "cpu",
        alpha: float = 2.3,
    ):
        super().__init__(
            input_size,
            output_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            use_regret_matching_argmax,
            device,
        )
        self.alpha = alpha
        self.reinitialize_imm_regret_networks = reinitialize_imm_regret_networks
        self.imm_model = MLP(self.input_size, self.network_layers, self.output_size).to(
            self.device
        )
        self.imm_optimizer = torch.optim.Adam(
            self.imm_model.parameters(), lr=self.learning_rate
        )

    def reset(self):
        super().reset()
        self.reset_immediate_regret()

    def reset_immediate_regret(self):
        if self.reinitialize_imm_regret_networks:
            self.imm_model.reset_parameters()
            self.imm_model.to(self.device)
            self.imm_optimizer = optim.Adam(
                self.imm_model.parameters(), lr=self.learning_rate
            )

    def train_model(self, T):
        # target_model and this replay buffer are fixed until the final sync.
        # Fit-local lifetime also invalidates the cache after reset/refill.
        cache = None
        if self.cache_frozen_predictions:
            cache = build_frozen_cache(
                min(len(self.buffer), self.buffer.buffer_size), self.batch_size,
                self.train_steps,
                lambda indices: self._regret_targets(self.buffer.gather(indices), T),
                device=self.device,
            )
        for train_step in range(self.train_steps):
            samples, indices = self.buffer.sample(self.batch_size, return_indices=True)
            targets = None if cache is None else cached_tensor(cache, indices, self.device)
            if train_step == 0 and cache is not None:
                direct = self._regret_targets(samples, T)
                if not torch.equal(targets, direct):
                    self.logger.warn("Frozen regret cache differs numerically; using direct inference")
                    cache = None
                    targets = direct
            loss = self.compute_loss(samples, T, targets=targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] regret loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )

            loss = self.compute_imm_loss(samples, T)
            self.imm_optimizer.zero_grad()
            loss.backward()
            self.imm_optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] imm regret loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )

        self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def _regret_targets(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, iterations = samples
        with torch.no_grad():
            target_outputs = self.predict(
                self.target_model, infostates, legal_actions_mask
            )
        zero = torch.zeros_like(target_outputs)
        target_outputs = torch.maximum(target_outputs, zero)
        return (
            target_outputs
            * math.pow(T - 1, self.alpha)
            / (math.pow(T - 1, self.alpha) + 1)
            + cf_regrets
        )

    def compute_loss(self, samples, T, *, targets=None):
        infostates, _, legal_actions_mask, _ = samples
        regrets = self._regret_targets(samples, T) if targets is None else targets
        outputs = self.predict(self.model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, regrets)
        return loss

    def compute_imm_loss(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, iterations = samples
        outputs = self.predict(self.imm_model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, cf_regrets)
        return loss

    def get_policy(self, s: SpielState, T: int, *, infostate=None,
                   legal_mask=None, legal_actions=None) -> np.ndarray:
        if infostate is None:
            infostate = self.get_infostate_tensor(s)
        if legal_mask is None:
            legal_mask = s.legal_actions_mask()
        if legal_actions is None:
            legal_actions = s.legal_actions()
        x = torch.as_tensor(infostate, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(legal_mask, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            regrets = self.predict(self.model, x, mask).cpu().numpy()
            imm_regrets = self.predict(self.imm_model, x, mask).cpu().numpy()
        return self.predictive_regret_matching(regrets, imm_regrets, legal_actions, T)

    def get_imm_regrets(self, s: SpielState) -> np.ndarray:
        imm_regrets = self.forward(
            self.imm_model, self.get_infostate_tensor(s), s.legal_actions_mask()
        )
        return imm_regrets

    def predictive_regret_matching(self, regrets, imm_regrets, legal_actions, T):
        key = (T, self.alpha)
        if getattr(self, "_discount_key", None) != key:
            self._discount_key = key
            self._discount_power = np.power(T - 1, self.alpha)
        predictive_regrets = np.maximum(
            np.maximum(regrets, 0)
            * self._discount_power
            / (self._discount_power + 1)
            + imm_regrets,
            0,
        )
        return self.regret_matching(predictive_regrets, legal_actions)


class VRPDCFRPlusQValueTrainer(QValueTrainer):
    cache_frozen_predictions = True

    def _next_strategies(self, next_states, next_legal_actions_mask, next_players, T):
        # Preserve the original arithmetic, player order and argmax tie rule.
        # Only regret networks are frozen for this fit, NOT target_model.
        players_next_strategies = []
        for player in [0, 1]:
            trainer = self.regret_trainers[player]
            regrets = trainer.predict(trainer.model, next_states, next_legal_actions_mask)
            imm_regrets = trainer.predict(trainer.imm_model, next_states, next_legal_actions_mask)
            power = np.power(T, trainer.alpha)
            regrets = torch.clamp(
                torch.clamp(regrets, min=0) * power / (power + 1) + imm_regrets, min=0,
            )
            legal_regrets = regrets * next_legal_actions_mask
            positive = torch.clamp(regrets, min=0)
            total = torch.sum(positive, dim=1, keepdim=True)
            matching = positive / total
            _, action = torch.max(torch.where(
                next_legal_actions_mask == 1, legal_regrets,
                torch.tensor(float("-inf"), device=self.device),
            ), dim=1)
            fallback = F.one_hot(action, self.output_size)
            players_next_strategies.append(torch.where(total == 0, fallback, matching))
        return torch.where(next_players.unsqueeze(1) == 0, *players_next_strategies)

    def _strategies_for_rows(self, indices, T):
        return self._next_strategies(
            self.buffer.numpy_to_float_tensor(self.buffer.next_states(indices)),
            self.buffer.numpy_to_int_tensor(self.buffer.next_legal_actions_mask_buf[indices]),
            self.buffer.numpy_to_int_tensor(self.buffer.next_player_buf[indices]), T,
        )

    def train_model(self, T):
        self.model = self.init_model()
        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.batch_size > 0 and len(self.buffer) < self.batch_size:
            return
        best_loss = float("inf")
        self.best_model = self.init_model()
        cache = None
        if self.cache_frozen_predictions:
            cache = build_frozen_cache(
                len(self.buffer), self.batch_size, self.train_steps + 1,
                lambda indices: self._strategies_for_rows(indices, T), device=self.device,
            )
        for train_step in range(self.train_steps + 1):
            samples, indices = self.buffer.sample(
                self.batch_size, return_indices=True, include_next_state=cache is None,
            )
            (
                histories,
                next_histories,
                next_states,
                rewards,
                next_legal_actions_mask,
                next_players,
                actions,
                dones,
            ) = samples

            q_values = self.model(histories)  # [B, A]
            # actions [B]
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

            with torch.no_grad():
                next_q_values = self.target_model(next_histories)  # [B, A]

                next_strategies = (
                    self._next_strategies(next_states, next_legal_actions_mask, next_players, T)
                    if cache is None else cached_tensor(cache, indices, self.device)
                )
                if train_step == 0 and cache is not None:
                    direct = self._strategies_for_rows(indices, T)
                    if not torch.equal(next_strategies, direct):
                        self.logger.warn("Frozen strategy cache differs numerically; using direct inference")
                        cache = None
                        next_strategies = direct

            target = rewards + (1 - dones) * torch.sum(
                next_q_values * next_strategies, dim=1, keepdim=False
            )

            loss = self.loss_fn(q_value, target)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if train_step % 50 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            loss_value = loss.item()
            if loss_value < best_loss:
                best_loss = loss_value
                self.best_model.load_state_dict(self.model.state_dict())

            if train_step % 100 == 0:
                self.logger.info(
                    "train_step[{}/{}]: loss {}, best_loss {}".format(
                        train_step, self.train_steps, loss_value, best_loss
                    )
                )

        self.model.load_state_dict(self.best_model.state_dict())
        return best_loss
