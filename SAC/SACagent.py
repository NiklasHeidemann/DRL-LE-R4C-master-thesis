from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

from SAC.ExperienceReplayBuffer import ExperienceReplayBuffer
from domain import ACTIONS
from loss_logger import LossLogger, ALPHA_VALUES


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
                 batch_size: int , model_path: str, target_entropy: float, seed: int):
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
        generator = tf.random.Generator.from_seed(seed=seed)
        self._generators = generator.split(count=len(agent_ids))

    def _get_max_q_value(self, states):
        reshaped_states = tf.reshape(np.array(states), shape=(
            1, self._environment.stats.recurrency, self._environment.stats.observation_dimension * len(self._agent_ids)))
        q = (tfm.minimum(self._critic_1(reshaped_states), self._critic_2(reshaped_states)))
        q_by_agent = tf.reshape(q, shape=(len(self._agent_ids), self._environment.stats.action_dimension))
        max_q_values = np.max(q_by_agent, axis=1)
        max_q_actions = np.argmax(q_by_agent, axis=1)
        return {agent_id: (max_q_value, max_q_action) for agent_id, max_q_value, max_q_action in
                zip(self._agent_ids, max_q_values, max_q_actions)}

    def save_models(self, name, run_name):
        if self._self_play:
            self._actor.save_weights(f"{self._model_path}{run_name}_actor{name}")
        else:
            for id, actor in self._actors.items():
                actor.save_weights(f"{self._model_path}{run_name}_actor_{id}_{name}")
        self._critic_1.save_weights(f"{self._model_path}{run_name}_critic_1{name}")
        self._critic_2.save_weights(f"{self._model_path}{run_name}_critic_2{name}")
        self._critic_1_t.save_weights(f"{self._model_path}{run_name}_critic_1_t{name}")
        self._critic_2_t.save_weights(f"{self._model_path}{run_name}_critic_2_t{name}")

    def load_models(self, name, run_name):
        if self._self_play:
            self._actor.load_weights(f"{self._model_path}{run_name}_actor{name}")
        else:
            for id, actor in self._actors.items():
                actor.load_weights(f"{self._model_path}{run_name}_actor_{id}_{name}")
        self._critic_1.load_weights(f"{self._model_path}{run_name}_critic_1{name}")
        self._critic_2.load_weights(f"{self._model_path}{run_name}_critic_2{name}")
        self._critic_1_t.load_weights(f"{self._model_path}{run_name}_critic_1_t{name}")
        self._critic_2_t.load_weights(f"{self._model_path}{run_name}_critic_2_t{name}")

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
        _, action_probs, log_probs = self.sample_actions(tf.reshape(states_prime,
                                                                                                    shape=(
                                                                                                    self._batch_size * len(
                                                                                                        self._agent_ids),
                                                                                                    self._environment.stats.recurrency,
                                                                                                    self._environment.stats.observation_dimension)),
                                                                                         actor=self._actor)
        flattened_states_prime = tf.reshape(states_prime, shape=self.agent_flattened_shape)
        flattened_states = tf.reshape(states, shape=self.agent_flattened_shape)
        q1 = tf.reshape(self._critic_1_t(flattened_states_prime), shape=action_probs.shape)
        q2 = tf.reshape(self._critic_2_t(flattened_states_prime), shape=action_probs.shape)
        q_r = tfm.minimum(q1, q2) - self._alpha * log_probs
        q_r_mean = tf.math.reduce_sum(action_probs * q_r, axis=1)
        targets = self._reward_scale * tf.reshape(rewards, q_r_mean.shape) + self._gamma * (
                1 - tf.repeat(dones,axis=0,repeats=len(self._agent_ids))) * q_r_mean
        loss_1, abs_11, abs_12 = self._critic_update(self._critic_1, flattened_states, actions, targets)
        loss_2, abs_21, abs_22 = self._critic_update(self._critic_2, flattened_states, actions, targets)
        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * log_probs, axis=1))
        return tf.add(loss_1, loss_2), tf.reduce_mean(tf.add(abs_11,abs_21)), tf.reduce_mean(tf.add(abs_12,abs_22)), log_probs, entropy

    def _critic_update(self, critic, states, actions, targets):
        with tf.GradientTape() as tape:
            q = tf.reduce_sum(
                tf.reshape(critic(states) * actions, shape=(self._batch_size * len(self._agent_ids), self._environment.stats.action_dimension)), axis=1)
            loss = 0.5 * self._mse(targets, q)
        gradients = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
        return loss, tf.math.abs(targets - q), tf.math.abs(q-targets)

    @tf.function
    def train_step_actor(self, states) -> Tuple[float, float, float]:
        reshaped_states = tf.reshape(states, shape=self.agent_flattened_shape)
        if self._self_play:
            actor = self._actor
            with tf.GradientTape() as tape:
                _, action_probs, log_probs = self.sample_actions(
                    states=tf.reshape(states, shape=(
                    self._batch_size * len(self._agent_ids), self._environment.stats.recurrency, self._environment.stats.observation_dimension)),
                    actor=actor)
                q = tf.reshape(tfm.minimum(self._critic_1(reshaped_states), self._critic_2(reshaped_states)),
                               shape=action_probs.shape)
                entropy_part = self._alpha * tf.concat([log_probs[:,:len(ACTIONS)],tf.zeros((log_probs.shape[0],log_probs.shape[1]-len(ACTIONS)))],axis=1)
                sum_part = tfm.reduce_sum(action_probs * (entropy_part - q), axis=1)
                loss = tfm.reduce_mean(sum_part)
            gradients = tape.gradient(loss, actor.trainable_variables)
            actor.optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
            return loss, tf.reduce_mean(q), tf.reduce_mean(tf.reduce_max(q, axis=1))
        else:
            raise NotImplementedError()


    @tf.function
    def sample_actions(self, states, actor, generator_index:int = 0, deterministic=False):
        probability_groups = actor(states) # first one for actions, rest for communication channels
        probability_groups = probability_groups if type(probability_groups) == list else [probability_groups]
        log_prob_groups = [tf.math.log(probabilities) for probabilities in probability_groups]
        if deterministic:
            action_groups = [tf.argmax(probabilities, axis=1) for probabilities in probability_groups]
        else:
            #action_groups = [self._generators[generator_index].categorical(logits=log_probs, num_samples=1)[:,0] for log_probs in log_prob_groups]
            action_groups = [tf.random.stateless_categorical(logits=log_probs, num_samples=1, seed=self._generators[generator_index].make_seeds(2)[0])[:,0] for log_probs in log_prob_groups]
        one_hot_action_groups = [tf.one_hot(actions, depth=probabilities.shape[-1]) for actions, probabilities in zip(action_groups,probability_groups)]
        return tf.squeeze(tf.concat(one_hot_action_groups,axis=-1)), tf.concat(probability_groups,axis=-1), tf.concat(log_prob_groups,axis=-1)


    def act_batched(self, batched_state, env_batcher,  deterministic:bool, multitimer=None) -> Tuple[Tuple, np.ndarray]:
        if multitimer is not None:
            multitimer.start("batch_compute_action_dict")
        batched_actions, batched_action_probs = self._wrap_batched_compute_actions_one_hot_and_prob(batched_state, deterministic)
        assert not tf.math.reduce_any(tf.math.is_nan(batched_action_probs)), f"{batched_action_probs}{batched_state}"
        if multitimer is not None:
            multitimer.stop("batch_compute_action_dict")
            multitimer.start("env_step")
        observation_prime, reward, done, truncated = env_batcher.step(batched_actions)
        if multitimer is not None:
            multitimer.stop("env_step")
        return (
            (batched_actions, observation_prime, reward, tf.math.logical_or(done, truncated)),batched_action_probs)
    def act(self, state, env,  deterministic:bool, multitimer=None) -> Tuple[Tuple, np.ndarray]:
        if multitimer is not None:
            multitimer.start("compute_action_dict")
        #actions, actions_array, all_actions = self._compute_actions(state, deterministic)
        actions, action_probs = self._compute_actions_one_hot_and_prob(state, deterministic)

        if multitimer is not None:
            multitimer.stop("compute_action_dict")
            multitimer.start("env_step")
        observation_prime, reward, done, truncated = env.step(actions)
        if multitimer is not None:
            multitimer.stop("env_step")
        return (
            (actions, observation_prime, reward, done or truncated),action_probs)


    def _wrap_batched_compute_actions_one_hot_and_prob(self, state, deterministic):
        actions_one_hot, actions_probs = self._batched_compute_actions_one_hot_and_prob(state, deterministic)
        if state.shape[0]==1:
            actions_one_hot, actions_probs = tf.expand_dims(actions_one_hot, axis=1), actions_probs
        return np.moveaxis(np.array(actions_one_hot), 0, 1), np.moveaxis(np.array(actions_probs), 0, 1)
    @tf.function
    def _batched_compute_actions_one_hot_and_prob(self, state, deterministic):
        actions_one_hot, actions_probs, _ = list(zip(*[self.sample_actions(deterministic=deterministic,generator_index=index,
                                                                           states= tf.convert_to_tensor(state[:,:,index,:]),
                                                                           actor=self._actor) for index in range(len(self._agent_ids))]))
        return actions_one_hot, actions_probs

    def _compute_actions_one_hot_and_prob(self, state, deterministic):
        assert state.shape == (self._environment.stats.recurrency, len(self._agent_ids), self._environment.stats.observation_dimension)
        actions_one_hot, actions_probs, _ = list(zip(*[self.sample_actions(deterministic=deterministic,generator_index=index,
                                                                           states=tf.expand_dims(
                                                                               tf.convert_to_tensor(state[:,index,:]), axis=0),
                                                                           actor=self._actor) for index in range(len(self._agent_ids))]))
        return np.array(actions_one_hot), np.squeeze(np.array(actions_probs))


    def train_step_temperature(self, states):
        action_probs = self._actor(tf.reshape(states, shape=(self._batch_size * len(self._agent_ids), self._environment.stats.recurrency, self._environment.stats.observation_dimension)))
        only_action_probs = action_probs[0] if type(action_probs) == list else action_probs # leave out communication
        log_probs = tf.math.log(only_action_probs)
        gradient = tf.reduce_mean(tf.reduce_sum(tf.multiply(only_action_probs, log_probs),axis=1))+self._target_entropy
        self._alpha += 0.1*self._learning_rate * gradient
        self._loss_logger.add_value(ALPHA_VALUES, self._alpha)

    @property
    def agent_flattened_shape(self):
        return (self._batch_size, self._environment.stats.recurrency, self._environment.stats.observation_dimension * len(self._agent_ids))

    def _get_actor(self, agent_id: str) -> tf.keras.Model:
        return self._actor if self._self_play else self._actors[agent_id]
