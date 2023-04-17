import datetime
from collections import defaultdict
from threading import Thread
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

from SAC.ExperienceReplayBuffer import RecurrentExperienceReplayBuffer, ExperienceReplayBuffer
from environment.render import render_permanently
from loss_logger import LossLogger, CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, RETURNS, Q_VALUES, MAX_Q_VALUES
from params import BATCH_SIZE, LEARNING_RATE, TIME_STEPS, TRAININGS_PER_TRAINING, ALPHA, ACTIONS, GAMMA
from plots import plot_multiple


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent
    Based on pseudocode of OpenAI Spinning Up (2022) (https://spinningup.openai.com/en/latest/algorithms/sac.html)
    and the Paper of Haarnoja et al. (2018) (https://arxiv.org/abs/1801.01290)

    :param environment: The environment to learn from (conforming to the gymnasium specifics https://gymnasium.farama.org/)
    :param state_dim: Dimensions of the state space
    :param action_dim: Dimensions of the action space
    :param actor_network_generator: a generator function for the actor network
        signature: learning_rate, state_dim, action_dim -> tensorflow Model
    :param critic_network_generator: a generator function for the critic networks (Q-networks)
        signature: learning_rate, state_dim, action_dim -> tensorflow Model
    :param learning_rate=0.0003: Learning rate for adam optimizer. # todo testing by kl-divergence
        The same learning rate will be used for all networks (Q-Values, Actor)
    :param gamma=0.99: discount factor
    :param tau=0.005:  Polyak averaging coefficient (between 0 and 1)
    :param alpha=0.2: Entropy regularization coefficient (between 0 and 1).
        Controlling the exploration/exploitation trade off.
        (inverse of reward scale in the original SAC paper)
    :param batch_size=256: Minibatch size for each gradient update
    :param max_replay_buffer_size=1000000: Size of the replay buffer
    :param model_path: path to the location the model is saved and loaded from
    """

    # todo cnn
    def __init__(self, environment, loss_logger: LossLogger, replay_buffer: ExperienceReplayBuffer, self_play: bool, agent_ids: List[str], action_dim,
                 actor_network_generator, critic_network_generator, recurrent: bool,
                 learning_rate: float, gamma: float, tau: float, reward_scale: float, alpha: float,
                 batch_size: int , model_path: str):
        self._environment = environment
        self._loss_logger = loss_logger
        self._self_play = self_play
        self._action_dim = action_dim
        self._gamma = gamma
        self._tau = tau
        self._reward_scale = reward_scale
        self._alpha = alpha
        self._batch_size = batch_size
        self._mse = tf.keras.losses.MeanSquaredError()
        self._model_path = model_path
        self._reply_buffer = replay_buffer
        self._agent_ids = agent_ids
        if self_play:
            self._actor = actor_network_generator(learning_rate, recurrent=recurrent)
        else:
            self._actors = {agent_id: actor_network_generator(learning_rate, recurrent=recurrent) for agent_id in
                            agent_ids}
        self._critic_1 = critic_network_generator(learning_rate=learning_rate, agent_num=len(agent_ids),
                                                  recurrent=recurrent)
        self._critic_2 = critic_network_generator(learning_rate=learning_rate, agent_num=len(agent_ids),
                                                  recurrent=recurrent)
        self._critic_1_t = critic_network_generator(learning_rate=learning_rate, agent_num=len(agent_ids),
                                                    recurrent=recurrent)
        self._critic_2_t = critic_network_generator(learning_rate=learning_rate, agent_num=len(agent_ids),
                                                    recurrent=recurrent)
        self._wight_init()

    def _get_max_q_value(self, states):
        reshaped_states = tf.reshape(np.array(list(states.values())), shape=(
            1, TIME_STEPS, self._environment.stats.observation_dimension * len(self._agent_ids)))
        q = (tfm.minimum(self._critic_1(reshaped_states), self._critic_2(reshaped_states)))
        q_by_agent = tf.reshape(q, shape=(len(self._agent_ids), len(ACTIONS)))
        max_q_values = np.max(q_by_agent, axis=1)
        max_q_actions = np.argmax(q_by_agent, axis=1)
        return {agent_id: (max_q_value, max_q_action) for agent_id, max_q_value, max_q_action in
                zip(states.keys(), max_q_values, max_q_actions)}

    def save_models(self, name):
        if self._self_play:
            self._actor.save_weights(f"{self._model_path}actor{name}")
        else:
            for id, actor in self._actors.items():
                actor.save_weights(f"{self._model_path}actor_{id}_{name}")
        self._critic_1.save_weights(f"{self._model_path}critic_1{name}")
        self._critic_2.save_weights(f"{self._model_path}critic_2{name}")
        self._critic_1_t.save_weights(f"{self._model_path}critic_1_t{name}")
        self._critic_2_t.save_weights(f"{self._model_path}critic_2_t{name}")

    def load_models(self, name):
        if self._self_play:
            self._actor.load_weights(f"{self._model_path}actor{name}")
        else:
            for id, actor in self._actors.items():
                actor.load_weights(f"{self._model_path}actor_{id}_{name}")
        self._critic_1.load_weights(f"{self._model_path}critic_1{name}")
        self._critic_2.load_weights(f"{self._model_path}critic_2{name}")
        self._critic_1_t.load_weights(f"{self._model_path}critic_1_t{name}")
        self._critic_2_t.load_weights(f"{self._model_path}critic_2_t{name}")

    def _wight_init(self):
        self._critic_1.set_weights(self._critic_1_t.weights)
        self._critic_2.set_weights(self._critic_2_t.weights)

    def update_target_weights(self):
        self._weight_update(self._critic_1_t, self._critic_1)
        self._weight_update(self._critic_2_t, self._critic_2)

    def _weight_update(self, target_network, network):
        new_wights = []
        for w_t, w in zip(target_network.weights, network.weights):
            new_wights.append((1 - self._tau) * w_t + self._tau * w)
        target_network.set_weights(new_wights)


    @tf.function
    def train_step_critic(self, states, actions, rewards, states_prime, dones):
        _, action_probs, log_probs = self.sample_actions_prime_and_log_probs_from_policy(tf.reshape(states_prime,
                                                                                                    shape=(
                                                                                                    BATCH_SIZE * len(
                                                                                                        self._agent_ids),
                                                                                                    TIME_STEPS,
                                                                                                    self._environment.stats.observation_dimension)),
                                                                                         actor=self._actor)
        flattened_states_prime = tf.reshape(states_prime, shape=self.agent_flattened_shape)
        flattened_states = tf.reshape(states, shape=self.agent_flattened_shape)
        q1 = tf.reshape(self._critic_1_t(flattened_states_prime), shape=action_probs.shape)
        q2 = tf.reshape(self._critic_2_t(flattened_states_prime), shape=action_probs.shape)
        q_r = tfm.minimum(q1, q2) - self._alpha * log_probs
        q_r_mean = tf.math.reduce_sum(action_probs * q_r, axis=1)
        targets = self._reward_scale * tf.reshape(rewards, q_r_mean.shape) + self._gamma * (
                1 - tf.reshape(dones, q_r_mean.shape)) * q_r_mean
        loss_1 = self._critic_update(self._critic_1, flattened_states, actions, targets)
        loss_2 = self._critic_update(self._critic_2, flattened_states, actions, targets)
        return tf.add(loss_1, loss_2), log_probs

    def _critic_update(self, critic, states, actions, targets):
        with tf.GradientTape() as tape:
            q = tf.reduce_sum(
                tf.reshape(critic(states) * actions, shape=(BATCH_SIZE * len(self._agent_ids), len(ACTIONS))), axis=1)
            loss = 0.5 * self._mse(targets, q)
        gradients = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
        return loss

    @tf.function
    def train_step_actor(self, states) -> Tuple[float, float, float]:
        reshaped_states = tf.reshape(states, shape=self.agent_flattened_shape)
        if self._self_play:
            actor = self._actor
            with tf.GradientTape() as tape:
                _, action_probs, log_probs = self.sample_actions_prime_and_log_probs_from_policy(
                    states=tf.reshape(states, shape=(
                    BATCH_SIZE * len(self._agent_ids), TIME_STEPS, self._environment.stats.observation_dimension)),
                    actor=actor)
                q = tf.reshape(tfm.minimum(self._critic_1(reshaped_states), self._critic_2(reshaped_states)),
                               shape=action_probs.shape)
                entropy_part = self._alpha * log_probs
                sum_part = tfm.reduce_sum(action_probs * (entropy_part - q), axis=1)
                loss = tfm.reduce_mean(sum_part)
            gradients = tape.gradient(loss, actor.trainable_variables)
            actor.optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
            return loss, tf.reduce_mean(q), tf.reduce_mean(tf.reduce_max(q, axis=1))
        else:
            raise NotImplementedError()

    @tf.function
    def sample_actions_prime_and_log_probs_from_policy(self, states, actor):
        probabilities = actor(states)
        log_probs = tf.math.log(probabilities)
        actions = tf.random.categorical(logits=log_probs, num_samples=1)[:, 0]
        one_hot_actions = tf.one_hot(actions, depth=len(ACTIONS))
        return one_hot_actions, probabilities, log_probs

    def act_deterministic(self, state):
        actions_prime = {}
        probabilities = {}
        for agent_id in self._agent_ids:
            actor = self._get_actor(agent_id=agent_id)
            agent_state = tf.convert_to_tensor([state[agent_id]], dtype=tf.float32)
            probabilities[agent_id] = actor(agent_state)[0]
            actions_prime[agent_id] = tf.reshape(
                tf.one_hot(
                    tf.argmax(probabilities[agent_id]), depth=len(ACTIONS)
                ),
                shape=(1, len(ACTIONS))
            )
        return self._act(actions_prime), probabilities

    def act_stochastic(self, state) -> Tuple[Tuple, Dict[str, np.ndarray]]:
        actions = {
            agent_id: self.sample_actions_prime_and_log_probs_from_policy(
                states=tf.expand_dims(tf.convert_to_tensor(states), axis=0), actor=self._actor) for agent_id, states in
            state.items()
        }
        return self._act({agent_id: actions[agent_id][0] for agent_id in self._agent_ids}), {
            agent_id: actions[agent_id][1] for agent_id in self._agent_ids}

    def _act(self, all_actions):
        communications = {agent_id: action[0][len(ACTIONS):] for agent_id, action in all_actions.items()}
        selected_communications = {agent_id: self._select_communication(communication=com) for agent_id, com in
                                   communications.items()}
        selected_actions = {agent_id: ACTIONS[np.argmax(action[0][:len(ACTIONS)])] for agent_id, action in
                            all_actions.items()}
        actions_dict = {agent_id: (selected_actions[agent_id], selected_communications[agent_id]) for agent_id in
                        all_actions.keys()}
        observation_prime, reward, done, truncated, _ = self._environment.step(actions_dict)
        return all_actions, observation_prime, reward, {agent_id: done[agent_id] or truncated[agent_id] for agent_id in
                                                        self._agent_ids}

    def _select_communication(self, communication: np.ndarray) -> np.ndarray:
        number_channels = self._environment.stats.number_communication_channels
        vocab_size = self._environment.stats.size_vocabulary
        base_array = np.zeros(shape=(number_channels * vocab_size))
        for base_index in range(0, len(base_array), vocab_size + 1):
            if communication[base_index] > 0:
                index = np.argmax(communication[base_index + 1:base_index + 2 + vocab_size])
                base_array[base_index + index] = 1
        return base_array


    @property
    def agent_flattened_shape(self):
        return (self._batch_size, TIME_STEPS, self._environment.stats.observation_dimension * len(self._agent_ids))

    def _get_actor(self, agent_id: str) -> tf.keras.Model:
        return self._actor if self._self_play else self._actors[agent_id]
