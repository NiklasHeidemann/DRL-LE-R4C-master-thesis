from abc import abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from scipy.special import kl_div
from typing_extensions import Protocol

from domain import ACTIONS
from environment.envbatcher import EnvBatcher
from training.ActionSampler import ActionSampler

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
              critic_network_generator, gamma: float, tau: float, mov_alpha: float, com_alpha: float,
              model_path: str, seed: int, social_influence_sample_size: int, social_reward_weight: float):
        self._agent_ids = agent_ids
        self._com_alpha = com_alpha
        self._mov_alpha = mov_alpha
        self._gamma = gamma
        self._model_path = model_path
        self._mse = tf.keras.losses.MeanSquaredError()
        self._social_influence_sample_size = social_influence_sample_size
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
                                             actor=self._actor)

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
                                                   deterministic=False) if include_social else 0.
        reward += self._social_reward_weight * social_reward
        return (
            (batched_actions, observation_prime, reward, tf.math.logical_or(done, truncated)), batched_action_probs,
            social_reward)

    def compute_social_reward(self, observation_prime: np.ndarray, deterministic: bool) -> np.ndarray:
        actual_probs = np.moveaxis(np.array(
            [self._actor(observation_prime[:, :, agent_index, :])[0] for agent_index in
             range(len(self._agent_ids))]), 0, 1)
        if deterministic:
            actual_probs = tf.one_hot(tf.argmax(actual_probs, axis=2), depth=actual_probs.shape[2])
        batched_additional_com_samples = self.random_additional_com_samples(stats=self._environment_batcher._stats, size=observation_prime.shape[0])
        mean_prob_alternatives = self.sample_additional_probs(observation_prime, batched_additional_com_samples, deterministic=deterministic)
        social_influence_pairs = np.array([[[(actual_probs[index, index_listener, :len(ACTIONS)],
                                              mean_prob_alternatives[
                                                  index, index_influencer, index_listener if index_listener < index_influencer else index_listener - 1])
                                             for index_listener in range(len(self._agent_ids)) if
                                             index_listener != index_influencer] for index_influencer in
                                            range(len(self._agent_ids))] for index in
                                           range(observation_prime.shape[0])])
        if deterministic:
            social_reward = [self._mse(social_influence_pairs[:, agent_index, :, 0], social_influence_pairs[:, agent_index, :, 1]) for agent_index in range(len(self._agent_ids))]
        else:
            exp_social_influence_pairs = np.exp(social_influence_pairs)
            kl_div_ = kl_div(exp_social_influence_pairs[:, :, :, 0], exp_social_influence_pairs[:, :, :, 1])
            kl_div_normalized = np.log(kl_div_ + 1)
            kl_div_by_agent = np.sum(kl_div_normalized, axis=2)
            social_reward = np.sum(kl_div_by_agent, axis=2)
        return social_reward


    def sample_additional_probs(self, states, additional_com_samples, deterministic: bool):
        repeated_states = np.array(
            tf.repeat(tf.expand_dims(states, axis=1), axis=1, repeats=additional_com_samples.shape[2]))
        mean_prob_original_action = np.zeros(
            shape=(self._environment_batcher.size, len(self._agent_ids), len(self._agent_ids) - 1, len(ACTIONS)))
        for index_influencer in range(states.shape[-2]):  # for each agent
            for index_listener in range(states.shape[-2]):  # for all other agents
                if index_listener == index_influencer:
                    continue
                pos_of_com = self._environment_batcher._stats.index_of_communication_in_observation(
                    agent_index=index_listener, speaker_index=index_influencer)
                repeated_states[:, :, -1, index_listener,
                pos_of_com:pos_of_com + additional_com_samples.shape[3]] = additional_com_samples[:, index_influencer]
                relevant_states = np.reshape(repeated_states[:, :, :, index_listener, :], newshape=(
                    repeated_states.shape[0] * repeated_states.shape[1], repeated_states.shape[2],
                    repeated_states.shape[4]))
                log_prob_actions = self._actor(relevant_states)[0]
                reshaped_log_prob_actions = tf.reshape(log_prob_actions, shape=(
                    repeated_states.shape[0], repeated_states.shape[1], len(ACTIONS)))
                if deterministic:
                    means_log_probs = tf.reduce_mean(
                        tf.one_hot(tf.argmax(reshaped_log_prob_actions, axis=2), depth=log_prob_actions.shape[1]),
                        axis=1)
                else:
                    means_log_probs = tf.reduce_mean(reshaped_log_prob_actions, axis=1)
                mean_prob_original_action[:, index_influencer,
                index_listener if index_listener < index_influencer else index_listener - 1, :] = means_log_probs
        return mean_prob_original_action

    def random_additional_com_samples(self, stats, size: int):
        logits = tf.math.log(tf.ones(shape=(
            size * self._social_influence_sample_size * stats.number_of_agents * stats.number_communication_channels,
            stats.size_vocabulary + 1)) / (stats.size_vocabulary + 1))
        samples = tf.random.stateless_categorical(logits=logits, num_samples=1,
                                                  seed=self._generators[0].make_seeds(2)[0])
        one_hot_samples = tf.one_hot(samples, depth=stats.size_vocabulary + 1)
        reshaped_samples = tf.reshape(one_hot_samples, shape=(
            size, stats.number_of_agents, self._social_influence_sample_size,
            stats.number_communication_channels * (stats.size_vocabulary + 1)))
        return reshaped_samples

    @property
    @abstractmethod
    def _environment_batcher(self):
        ...
