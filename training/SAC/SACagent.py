from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm
from typing_extensions import override

from domain import ACTIONS
from environment.envbatcher import EnvBatcher
from utils.loss_logger import LOG_PROBS, CRITIC_LOSS, ENTROPY, COM_ENTROPY, Q_VALUES, ACTOR_LOSS, \
    MAX_Q_VALUES
from training.Agent import Agent, EarlyStopping
from training.ExperienceReplayBuffer import STATE_KEY, ACTION_KEY, REWARD_KEY
from training.SAC.ExperienceReplayBuffer import DONE_KEY, STATE_PRIME_KEY


class SACAgent(Agent):

    def __init__(self, environment, agent_ids: List[str], actor_network_generator,
                 critic_network_generator, env_batcher: EnvBatcher,
                 gamma: float, tau: float, mov_alpha: float, com_alpha: float,
                 social_influence_sample_size: int,
                 batch_size: int, model_path: str, target_entropy: float, seed: int, social_reward_weight: float):
        self._environment = environment
        self._environment_batcher_ = env_batcher
        self._batch_size = batch_size
        self._target_entropy = target_entropy
        self._init(agent_ids=agent_ids, actor_network_generator=actor_network_generator,
                   actor_uses_log_probs=False, social_reward_weight=social_reward_weight, social_influence_sample_size=social_influence_sample_size,
                   critic_network_generator=critic_network_generator,
                   gamma=gamma, tau=tau, mov_alpha=mov_alpha, com_alpha=com_alpha, model_path=model_path, seed=seed)

    def _get_max_q_value(self, states):
        reshaped_states = tf.reshape(np.array(states), shape=(
            1, self._environment.stats.recurrency,
            self._environment.stats.observation_dimension * len(self._agent_ids)))
        q = (tfm.minimum(self._critic_1(reshaped_states), self._critic_2(reshaped_states)))
        q_by_agent = tf.reshape(q, shape=(len(self._agent_ids), self._environment.stats.action_dimension))
        max_q_values = np.max(q_by_agent, axis=1)
        max_q_actions = np.argmax(q_by_agent, axis=1)
        return {agent_id: (max_q_value, max_q_action) for agent_id, max_q_value, max_q_action in
                zip(self._agent_ids, max_q_values, max_q_actions)}

    @override
    @tf.function
    def train_step_critic(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, float]:
        states, actions, rewards, states_prime, dones = inputs[STATE_KEY], inputs[ACTION_KEY], inputs[REWARD_KEY], \
            inputs[STATE_PRIME_KEY], inputs[DONE_KEY]
        _, log_probs, action_probs = self._action_sampler(tf.reshape(states_prime,
                                                                    shape=(
                                                                        self._batch_size * len(
                                                                            self._agent_ids),
                                                                        self._environment.stats.recurrency,
                                                                        self._environment.stats.observation_dimension)),
                                                         generator_index=0)
        # [self._action_sampler(states=states_prime[:,index,:,:],                                                         actor=self._actors[index],generator_index=index) for index in range(len(self._actors))]
        flattened_states_prime = tf.reshape(states_prime, shape=self.agent_flattened_shape)
        flattened_states = tf.reshape(states, shape=self.agent_flattened_shape)
        q1 = tf.reshape(self._critic_1_t(flattened_states_prime), shape=action_probs.shape)
        q2 = tf.reshape(self._critic_2_t(flattened_states_prime), shape=action_probs.shape)
        entropy_part = tf.concat(
            [self._mov_alpha * log_probs[:, :len(ACTIONS)], self._com_alpha * log_probs[:, len(ACTIONS):]], axis=1)
        q_r = tfm.minimum(q1, q2) - entropy_part
        q_r_mean = tf.math.reduce_sum(action_probs * q_r, axis=1)
        targets = tf.reshape(rewards, q_r_mean.shape) + self._gamma * (
                1 - tf.repeat(dones, axis=0, repeats=len(self._agent_ids))) * q_r_mean
        loss_1, abs_11, abs_12 = self._critic_update(self._critic_1, flattened_states, actions, targets)
        loss_2, abs_21, abs_22 = self._critic_update(self._critic_2, flattened_states, actions, targets)
        entropy = -tf.reduce_mean(tf.reduce_sum((action_probs * log_probs)[:, :len(ACTIONS)], axis=1))
        com_entropy = -tf.reduce_mean(tf.reduce_sum((action_probs * log_probs)[:, len(ACTIONS):], axis=1))
        metrics = {CRITIC_LOSS: tf.add(loss_1, loss_2), LOG_PROBS: log_probs, ENTROPY: entropy,
                   COM_ENTROPY: com_entropy}
        return metrics

    def _critic_update(self, critic, states, actions, targets):
        with tf.GradientTape() as tape:
            q = tf.reduce_sum(
                tf.reshape(critic(states) * actions,
                           shape=(self._batch_size * len(self._agent_ids), self._environment.stats.action_dimension)),
                axis=1)
            loss = 0.5 * self._mse(targets, q)
        gradients = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
        return loss, tf.math.abs(targets - q), tf.math.abs(q - targets)

    @override
    @tf.function
    def train_step_actor(self, batch: Dict[str, tf.Tensor]) -> Tuple[EarlyStopping, Dict[str, float]]:
        states = batch[STATE_KEY]
        reshaped_states = tf.reshape(states, shape=self.agent_flattened_shape)
        with tf.GradientTape() as tape:
            _, log_probs, action_probs = self._action_sampler(
                states=tf.reshape(states, shape=(
                    self._batch_size * len(self._agent_ids), self._environment.stats.recurrency,
                    self._environment.stats.observation_dimension)),
                generator_index=0)
            q = tf.reshape(tfm.minimum(self._critic_1(reshaped_states), self._critic_2(reshaped_states)),
                           shape=action_probs.shape)
            entropy_part = tf.concat(
                [self._mov_alpha * log_probs[:, :len(ACTIONS)], self._com_alpha * log_probs[:, len(ACTIONS):]], axis=1)
            sum_part = tfm.reduce_sum(action_probs * (entropy_part - q), axis=1)
            loss = tfm.reduce_mean(sum_part)
        gradients = tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(gradients, self._actor.trainable_variables))
        metrics = {ACTOR_LOSS: loss, Q_VALUES: tf.reduce_mean(q),
                   MAX_Q_VALUES: tf.reduce_mean(tf.reduce_max(q, axis=1))}
        return False, metrics

    @tf.function
    def train_step_single_actor(self, index: int,agent_states:tf.Tensor,q:tf.Tensor)->Dict[str,float]:
            with tf.GradientTape() as tape:
                _, log_probs, action_probs = self._action_sampler(
                    states=tf.reshape(agent_states, shape=(
                        self._batch_size, self._environment.stats.recurrency,
                        self._environment.stats.observation_dimension)),
                generator_index=index)
                entropy_part = tf.concat(
                    [self._mov_alpha * log_probs[:, :len(ACTIONS)], self._com_alpha * log_probs[:, len(ACTIONS):]], axis=1)
                sum_part = tfm.reduce_sum(action_probs * (entropy_part - q), axis=1)
                loss = tfm.reduce_mean(sum_part)
            gradients = tape.gradient(loss, self._actor.trainable_variables)
            self._actor.optimizer.apply_gradients(zip(gradients, self._actor.trainable_variables))
            metrics = {ACTOR_LOSS: loss, Q_VALUES: tf.reduce_mean(q),
                       MAX_Q_VALUES: tf.reduce_mean(tf.reduce_max(q, axis=1))}
            return metrics

    def train_step_temperature(self, states):
        action_probs = self._actor(tf.reshape(states, shape=(
        self._batch_size * len(self._agent_ids), self._environment.stats.recurrency,
        self._environment.stats.observation_dimension)))
        only_action_probs = action_probs[0] if type(action_probs) == list else action_probs  # leave out communication
        log_probs = tf.math.log(only_action_probs)
        gradient = tf.reduce_mean(
            tf.reduce_sum(tf.multiply(only_action_probs, log_probs), axis=1)) + self._target_entropy
        self._mov_alpha += 0.1 * self._learning_rate * gradient

    @property
    def agent_flattened_shape(self):
        return (self._batch_size, self._environment.stats.recurrency,
                self._environment.stats.observation_dimension * len(self._agent_ids))

    @property
    def _environment_batcher(self)->EnvBatcher:
        return self._environment_batcher_
