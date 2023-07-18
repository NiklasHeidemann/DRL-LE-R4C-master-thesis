from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

from SAC.ExperienceReplayBuffer import ExperienceReplayBuffer
from domain import ACTIONS
from loss_logger import LossLogger, ALPHA_VALUES
from tensorflow_probability import distributions as tfp

class PPOAgent:

    #todo
    def __init__(self, environment_batcher, loss_logger: LossLogger, replay_buffer: ExperienceReplayBuffer, self_play: bool, agent_ids: List[str], action_dim,
                 actor_network_generator, critic_network_generator, recurrent: bool, epsilon: float,
                 learning_rate: float, gamma: float, tau: float, reward_scale: float, alpha: float, l_alpha: float,
                 batch_size: int , model_path: str, target_entropy: float, seed: int, gae_lamda: float, kld_threshold: float):
        self._environment_batcher = environment_batcher
        self._loss_logger = loss_logger
        self._self_play = self_play
        self._action_dim = action_dim
        self._epsilon = epsilon
        self._gamma = gamma
        self._tau = tau
        self._kld_threshold = kld_threshold
        self._reward_scale = reward_scale
        self._alpha = alpha
        self._l_alpha = l_alpha
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._target_entropy = target_entropy
        self._mse = tf.keras.losses.MeanSquaredError()
        self._model_path = model_path
        self._reply_buffer = replay_buffer
        self._agent_ids = agent_ids
        self._gae_lamda = gae_lamda
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

    #todo
    def _get_max_q_value(self, states):
        reshaped_states = tf.reshape(np.array(states), shape=(
            1, self._environment_batcher._stats.recurrency, self._environment_batcher._stats.observation_dimension * len(self._agent_ids)))
        q = (tfm.minimum(self._critic_1(reshaped_states), self._critic_2(reshaped_states)))
        q_by_agent = tf.reshape(q, shape=(len(self._agent_ids), self._environment_batcher._stats.action_dimension))
        max_q_values = np.max(q_by_agent, axis=1)
        max_q_actions = np.argmax(q_by_agent, axis=1)
        return {agent_id: (max_q_value, max_q_action) for agent_id, max_q_value, max_q_action in
                zip(self._agent_ids, max_q_values, max_q_actions)}

    #todo
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

    #todo
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

    #todo
    def _wight_init(self):
        self._critic_1.set_weights(self._critic_1_t.weights)
        self._critic_2.set_weights(self._critic_2_t.weights)

    #todo
    def update_target_weights(self):
        self._weight_update(self._critic_1_t, self._critic_1)
        self._weight_update(self._critic_2_t, self._critic_2)

    #todo
    def _weight_update(self, target_network, network):
        new_wights = []
        for w_t, w in zip(target_network.weights, network.weights):
            new_wights.append((1 - self._tau) * w_t + self._tau * w)
        target_network.set_weights(new_wights)


    #@tf.function
    def train_step_critic(self, states, ret):
        reshaped_states = tf.reshape(states, shape=(states.shape[0], self._environment_batcher._stats.recurrency, self._environment_batcher._stats.observation_dimension * len(self._agent_ids)))
        losses = 0
        for critic in [self._critic_1, self._critic_2]:
            with tf.GradientTape() as tape:
                prev_v = critic(reshaped_states)
                loss = self._mse(ret, prev_v)
            gradients = tape.gradient(loss, critic.trainable_variables)
            critic.optimizer.apply_gradients(
                zip(gradients, critic.trainable_variables)
            )
            losses+=loss
        return losses, tf.reduce_mean(prev_v)

    #@tf.function
    def log_probs_and_entropy_from_policy(self, state, action):
        probability_groups = self._actor(state)
        probability_groups = probability_groups if type(probability_groups) == list else [probability_groups]
        distributions = [tfp.Categorical(probs=probability_group) for probability_group in probability_groups]
        log_probs = [distribution.log_prob(tf.argmax(action,axis=1)) for distribution in distributions]
        entropy = [-tf.reduce_sum(probability_group * tf.math.log(tf.clip_by_value(probability_group, 1e-10, 10)), axis=1) for
         probability_group in probability_groups]
        if sum([True in tf.math.is_nan(entropy_group) for entropy_group in entropy]) > 0:
            raise Exception(
                f"{sum([tf.reduce_sum(tf.cast(tf.math.is_nan(entropy_group), tf.float32)) for entropy_group in probability_groups])}\n{sum([True in tf.math.is_nan(entropy_group) for entropy_group in log_probs])}\n{tf.reduce_sum(tf.cast(tf.math.is_nan(state),tf.float32))}\n{entropy}")
        return log_probs, entropy


    #@tf.function
    def train_step_actor(self, state, action, advantage, prob_old):
        if self._self_play:
            actor = self._actor
            with tf.GradientTape() as tape:
                log_prob_current, entropy = self.log_probs_and_entropy_from_policy(state, action)
                p = tf.math.exp(log_prob_current - prob_old)  # exp() to un do log(p)
                clipped_p = tf.clip_by_value(p, 1 - self._epsilon, 1 + self._epsilon)
                policy_loss = -tfm.reduce_mean(tfm.minimum(p * advantage, clipped_p * advantage))
                entropy_loss = -tfm.reduce_mean(entropy)
                loss = policy_loss + self._alpha * entropy_loss
            gradients = tape.gradient(loss, actor.trainable_variables)
            # check whether actor weights contain nan
            if sum([tf.reduce_sum(tf.cast(tf.math.is_nan(weight),tf.float32)) for weight in gradients]) > 0:
                raise Exception(f"Actor weights contain nan, {loss} ::: {gradients}")
            actor.optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
            log_ratio = prob_old - log_prob_current
            kld = tf.math.reduce_mean((tf.math.exp(log_ratio) - 1) - log_ratio)
            early_stopping = kld > self._kld_threshold
            return loss, early_stopping, entropy_loss, kld
        else:
            raise NotImplementedError()


    #@tf.function
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
        # check whether probabilitygroups contains nan
        if sum([tf.reduce_sum(tf.cast(tf.math.is_nan(probability_group),tf.float32)) for probability_group in probability_groups]) > 0:
            raise Exception(f"{sum([tf.reduce_sum(tf.cast(tf.math.is_nan(probability_group),tf.float32)) for probability_group in probability_groups])}\n{tf.reduce_sum(tf.cast(tf.math.is_nan(states),tf.float32))}{probability_groups}")
        return tf.squeeze(tf.concat(one_hot_action_groups,axis=-1)), tf.concat(probability_groups,axis=-1), tf.concat(log_prob_groups,axis=-1)


    def act_batched(self, batched_state, env_batcher,  deterministic:bool, multitimer=None) -> Tuple[Tuple, np.ndarray]:
        if multitimer is not None:
            multitimer.start("batch_compute_action_dict")
        batched_actions, batched_action_probs = self._wrap_batched_compute_actions_one_hot_and_log_prob(batched_state, deterministic)
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
        actions, log_probs = self._compute_actions_one_hot_and_log_prob(state, deterministic)

        observation_prime, reward, done, truncated = env.step(actions)
        return (
            (actions, observation_prime, reward, done or truncated),log_probs)


    def _wrap_batched_compute_actions_one_hot_and_log_prob(self, state, deterministic):
        actions_one_hot, log_probs = self._batched_compute_actions_one_hot_and_log_prob(state, deterministic)
        if state.shape[0]==1:
            actions_one_hot, log_probs = tf.expand_dims(actions_one_hot, axis=1), log_probs
        return np.moveaxis(np.array(actions_one_hot), 0, 1), np.moveaxis(np.array(log_probs), 0, 1)


    #@tf.function
    def _batched_compute_actions_one_hot_and_log_prob(self, state, deterministic):
        actions_one_hot, _, log_probs = list(zip(*[self.sample_actions(deterministic=deterministic,generator_index=index,
                                                                           states= tf.convert_to_tensor(state[:,:,index,:]),
                                                                           actor=self._actor) for index in range(len(self._agent_ids))]))
        return actions_one_hot, log_probs


    def _compute_actions_one_hot_and_log_prob(self, state, deterministic):
        assert state.shape == (self._environment_batcher._stats.recurrency, len(self._agent_ids), self._environment_batcher._stats.observation_dimension)
        actions_one_hot, _, log_probs = list(zip(*[self.sample_actions(deterministic=deterministic,generator_index=index,
                                                                           states=tf.expand_dims(
                                                                               tf.convert_to_tensor(state[:,index,:]), axis=0),
                                                                           actor=self._actor) for index in range(len(self._agent_ids))]))
        return np.array(actions_one_hot), np.squeeze(np.array(log_probs))


    def train_step_temperature(self, states):
        raise NotImplementedError()

    @property
    def agent_flattened_shape(self):
        return (self._batch_size, self._environment_batcher._stats.recurrency, self._environment_batcher._stats.observation_dimension * len(self._agent_ids))

    def _get_actor(self, agent_id: str) -> tf.keras.Model:
        return self._actor if self._self_play else self._actors[agent_id]

    # generalized advantage estimate (taken from https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/)
    def estimate_advantage(self, rewards, values, dones, next_done, next_value):
        adv = np.zeros_like(rewards)
        last_gae_lamda = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - int(next_done)
                next_values = next_value
            else:
                next_non_terminal = 1.0 - int(dones[t + 1])
                next_values = values[t + 1]
            delta = rewards[t] + self._gamma * next_values * next_non_terminal - values[t]
            adv[t] = last_gae_lamda = delta + self._gamma * self._gae_lamda * next_non_terminal * last_gae_lamda
        return adv

    def get_values(self, states):
        reshaped_states = tf.reshape(states, (states.shape[0], self._environment_batcher._stats.recurrency, self._environment_batcher._stats.observation_dimension * len(self._agent_ids)))
        v1 = self._critic_1(reshaped_states)
        v2 = self._critic_2(reshaped_states)
        return tf.minimum(v1, v2)

    def sample_trajectories(self, steps_per_trajectory):
        observation = self._environment_batcher.reset_all()
        resets = 1
        rewards = []
        dones = []
        values = []
        observations = []
        actions = []
        probabilities = []
        last_done = [False] * self._environment_batcher.size

        for _ in range(steps_per_trajectory//self._environment_batcher.size):
            (action, new_observation, reward, next_done), log_probs = self.act_batched(observation, self._environment_batcher, deterministic=False)
            rewards.append(reward)
            values.append(self.get_values(states=observation))
            dones.append(last_done)
            observations.append(observation)
            actions.append(action)
            action_indexes = tf.argmax(action, axis=2)
            probabilities.append([(log_probs[index][:,action_indexes[index]].diagonal()) for index in range(log_probs.shape[0])])
            observation = self._environment_batcher.reset(mask=next_done, observation_array=new_observation)
            last_done = next_done
        next_value = self.get_values(states=observation)
        advantages_list = [self.estimate_advantage(rewards=reward_batch, values=value_batch, dones=done_batch, next_done=next_done_batch, next_value=next_value_batch) for reward_batch, value_batch, done_batch, next_done_batch, next_value_batch in zip(rewards, values, dones, next_done, next_value)]
        return (tf.convert_to_tensor([element for batch in observations for element in batch]), tf.convert_to_tensor([element for batch in actions for element in batch]),
                                                 tf.convert_to_tensor([element for batch in advantages_list for element in batch]),
                                                 tf.convert_to_tensor([element for batch in rewards for element in batch]),
                                                 tf.convert_to_tensor([element for batch in probabilities for element in batch]))