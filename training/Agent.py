from abc import abstractmethod
from typing import List, Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from scipy.special import kl_div
from typing_extensions import Protocol

from domain import ACTIONS
from environment.envbatcher import EnvBatcher
from training.ActionSampler import ActionSampler
from training.SocialRewardComputer import SocialRewardComputer

EarlyStopping = bool


class Agent(Protocol):
    """
    An agent that can be trained by a trainer.

    agent_ids: The ids of the agents.
    actor_network_generator: A function that returns an actor network as tensorflow model.
    critic_network_generator: A function that returns a critic network as tensorflow model.
    recurrent: Whether the networks are recurrent or not, that is use an LSTM-layer.
    learning_rate: The learning rate for the actor and critic networks.
    gamma: The reward discount factor.
    tau: The soft update factor for the target networks.
    alpha: The entropy regularization factor for the moving actions.
    com_alpha: The entropy regularization factor for the communication actions.
    model_path: The path where the models are saved to and loaded from.
    seed: The seed for the random number generator.
    """

    def _init(self, agent_ids: List[str], actor_network_generator, actor_uses_log_probs: bool,
              critic_network_generator, gamma: float, tau: float, mov_alpha: float, com_alpha: float,epsilon:Optional[float],
              model_path: str, seed: int, social_influence_sample_size: int, social_reward_weight: float):
        self._agent_ids = agent_ids
        self._com_alpha = com_alpha
        self._mov_alpha = mov_alpha
        self._gamma = gamma
        self._model_path = model_path
        self._mse = tf.keras.losses.MeanSquaredError()
        self._social_reward_weight = social_reward_weight
        self._tau = tau

        self._actor = actor_network_generator()
        self._critic_1 = critic_network_generator()
        self._critic_2 = critic_network_generator()
        self._critic_1_t = critic_network_generator()
        self._critic_2_t = critic_network_generator()
        self._weight_init()
        self._generators = tf.random.Generator.from_seed(seed=seed).split(count=len(agent_ids))
        self._action_sampler = ActionSampler(generators=self._generators, actor_uses_log_probs=actor_uses_log_probs,
                                             actor=self._actor, epsilon=epsilon)
        self._social_reward_computer = SocialRewardComputer(generators=self._generators,num_agents=len(agent_ids),social_influence_sample_size=social_influence_sample_size,
                                                            stats=self._environment_batcher._stats)

    def _weight_init(self):
        self._critic_1.set_weights(self._critic_1_t.weights)
        self._critic_2.set_weights(self._critic_2_t.weights)

    def save_models(self, name: str, run_name: str, current_epoch: int):
        self._actor.save_weights(f"{self._model_path}{run_name}_actor{name}")
        self._critic_1.save_weights(f"{self._model_path}{run_name}_critic_1{name}")
        self._critic_2.save_weights(f"{self._model_path}{run_name}_critic_2{name}")
        self._critic_1_t.save_weights(f"{self._model_path}{run_name}_critic_1_t{name}")
        self._critic_2_t.save_weights(f"{self._model_path}{run_name}_critic_2_t{name}")
        with open(f"{self._model_path}{run_name}_epoch{name}", "w") as file:
            file.write(str(current_epoch))

    def load_models(self, name, run_name):
        self._actor.load_weights(f"{self._model_path}{run_name}_actor{name}")
        self._critic_1.load_weights(f"{self._model_path}{run_name}_critic_1{name}")
        self._critic_2.load_weights(f"{self._model_path}{run_name}_critic_2{name}")
        self._critic_1_t.load_weights(f"{self._model_path}{run_name}_critic_1_t{name}")
        self._critic_2_t.load_weights(f"{self._model_path}{run_name}_critic_2_t{name}")
        with open(f"{self._model_path}{run_name}_epoch{name}", "r") as file:
            current_epoch = int(file.read())
        return current_epoch

    def update_target_weights(self):
        self._weight_update(self._critic_1_t, self._critic_1)
        self._weight_update(self._critic_2_t, self._critic_2)

    def _weight_update(self, target_network, network):
        new_weights = []
        for w_t, w in zip(target_network.weights, network.weights):
            new_weights.append((1 - self._tau) * w_t + self._tau * w)
        target_network.set_weights(new_weights)

    @abstractmethod
    def train_step_critic(self, batch: Dict[str, tf.Tensor]) -> Dict[str, float]:
        ...

    @abstractmethod
    def train_step_actor(self, batch: Dict[str, tf.Tensor]) -> Tuple[EarlyStopping, Dict[str, float]]:
        ...

    def act(self, state, env, deterministic: bool) -> Tuple[Tuple, np.ndarray]:
        actions, action_probs_or_log_probs = self._action_sampler.compute_actions_one_hot_and_prob(state=state,
                                                                                                   deterministic=deterministic)
        observation_prime, reward, done, truncated, _ = env.step(actions)
        return (
            (actions, observation_prime, reward, done or truncated), action_probs_or_log_probs)

    def act_batched(self, batched_state: tf.Tensor, env_batcher: EnvBatcher, deterministic: bool,
                    include_social: bool, ) -> Tuple[
        Tuple, np.ndarray, np.ndarray]:
        batched_actions, batched_action_probs = self._action_sampler.batched_compute_actions_one_hot_and_probs_or_log_probs(
            state=batched_state, deterministic=deterministic)
        observation_prime, reward, done, truncated = env_batcher.step(batched_actions)
        social_reward = self.compute_social_reward(observation_prime=observation_prime,
                                                   deterministic=False) if include_social and self._environment_batcher._stats.number_communication_channels>0 else 0.
        reward += self._social_reward_weight * social_reward
        return (
            (batched_actions, observation_prime, reward, tf.math.logical_or(done, truncated)), batched_action_probs,
            social_reward)

    def compute_social_reward(self, observation_prime: tf.Tensor, deterministic: bool) -> np.ndarray:
        return self._social_reward_computer(observation_prime=observation_prime,
                                                                  deterministic=deterministic,actor=self._actor)

    @abstractmethod
    def get_values(self, states: tf.Tensor)->tf.Tensor:
        ...
    @property
    @abstractmethod
    def _environment_batcher(self):
        ...
