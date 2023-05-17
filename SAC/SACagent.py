from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

from SAC.ExperienceReplayBuffer import ExperienceReplayBuffer
from loss_logger import LossLogger, ALPHA_VALUES
from params import ACTIONS, TIME_STEPS, SIZE_VOCABULARY


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
    :param learning_rate=0.0003: Learning rate for adam optimizer.
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

    def __init__(self, environment, loss_logger: LossLogger, replay_buffer: ExperienceReplayBuffer, self_play: bool, agent_ids: List[str], action_dim,
                 actor_network_generator, critic_network_generator, recurrent: bool,
                 learning_rate: float, gamma: float, tau: float, reward_scale: float, alpha: float,
                 batch_size: int , model_path: str, target_entropy: float):
        self._environment = environment
        self._loss_logger = loss_logger
        self._self_play = self_play
        self._action_dim = action_dim
        self._gamma = gamma
        self._tau = tau
        self._reward_scale = reward_scale
        self._alpha = alpha
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._target_entropy = target_entropy
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
        q_by_agent = tf.reshape(q, shape=(len(self._agent_ids), self._environment.stats.action_dimension))
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


    #@tf.function
    def train_step_critic(self, states, actions, rewards, states_prime, dones):
        _, action_probs, log_probs = self.sample_actions(tf.reshape(states_prime,
                                                                                                    shape=(
                                                                                                    self._batch_size * len(
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
        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * log_probs, axis=1))
        return tf.add(loss_1, loss_2), log_probs, entropy

    def _critic_update(self, critic, states, actions, targets):
        with tf.GradientTape() as tape:
            q = tf.reduce_sum(
                tf.reshape(critic(states) * actions, shape=(self._batch_size * len(self._agent_ids), self._environment.stats.action_dimension)), axis=1)
            loss = 0.5 * self._mse(targets, q)
        gradients = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
        return loss

    #@tf.function
    def train_step_actor(self, states) -> Tuple[float, float, float]:
        reshaped_states = tf.reshape(states, shape=self.agent_flattened_shape)
        if self._self_play:
            actor = self._actor
            with tf.GradientTape() as tape:
                _, action_probs, log_probs = self.sample_actions(
                    states=tf.reshape(states, shape=(
                    self._batch_size * len(self._agent_ids), TIME_STEPS, self._environment.stats.observation_dimension)),
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


    #@tf.function
    def sample_actions(self, states, actor, deterministic=False):
        probability_groups = actor(states) # first one for actions, rest for communication channels
        probability_groups = probability_groups if type(probability_groups) == list else [probability_groups]
        log_prob_groups = [tf.math.log(probabilities) for probabilities in probability_groups]
        if deterministic:
            action_groups = [tf.argmax(probabilities, axis=-1)[0] for probabilities in probability_groups]
        else:
            action_groups = [tf.random.categorical(logits=log_probs, num_samples=1)[0,0] for log_probs in log_prob_groups]
        one_hot_action_groups = [tf.one_hot(actions, depth=probabilities.shape[-1]) for actions, probabilities in zip(action_groups,probability_groups)]
        return tf.concat(one_hot_action_groups,axis=-1), tf.concat(probability_groups,axis=-1), tf.concat(log_prob_groups,axis=-1)



    def act(self, state, env,  deterministic:bool, multitimer=None) -> Tuple[Tuple, Dict[str, np.ndarray]]:
        if multitimer is not None:
            multitimer.start("tf")
        actions = {
            agent_id: self.sample_actions(deterministic=deterministic,
                states=tf.expand_dims(tf.convert_to_tensor(states), axis=0), actor=self._actor) for agent_id, states in
            state.items()
        }
        if multitimer is not None:
            multitimer.stop("tf")
            multitimer.start("env")
        a = self._act({agent_id: actions[agent_id][0] for agent_id in self._agent_ids},env=env), {
            agent_id: actions[agent_id][1] for agent_id in self._agent_ids}
        if multitimer is not None:
            multitimer.stop("env")
        return a

    def _act(self, all_actions, env):
        communications = {agent_id: tf.squeeze(action[len(ACTIONS):]) for agent_id, action in all_actions.items()}
        selected_actions = {agent_id: ACTIONS[np.argmax(action[:len(ACTIONS)])] for agent_id, action in
                            all_actions.items()}
        actions_dict = {agent_id: (selected_actions[agent_id], communications[agent_id]) for agent_id in
                        all_actions.keys()}
        observation_prime, reward, done, truncated, _ = env.step(actions_dict)
        return all_actions, observation_prime, reward, {agent_id: done[agent_id] or truncated[agent_id] for agent_id in
                                                        self._agent_ids}

    def train_step_temperature(self, states):
        action_probs = self._actor(tf.reshape(states, shape=(self._batch_size * len(self._agent_ids), TIME_STEPS, self._environment.stats.observation_dimension)))
        only_action_probs = action_probs[0] if type(action_probs) == list else action_probs # leave out communication
        log_probs = tf.math.log(only_action_probs)
        gradient = tf.reduce_mean(
            tf.matmul(
                tf.reshape(only_action_probs, shape=(256, 1, 5)),
                tf.reshape(self._alpha * log_probs, shape=(256, 5, 1))
            ) + self._target_entropy
        )
        #self._alpha -= self._learning_rate * gradient
        self._loss_logger.add_value(ALPHA_VALUES, self._alpha)

    @property
    def agent_flattened_shape(self):
        return (self._batch_size, TIME_STEPS, self._environment.stats.observation_dimension * len(self._agent_ids))

    def _get_actor(self, agent_id: str) -> tf.keras.Model:
        return self._actor if self._self_play else self._actors[agent_id]
