from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

from SAC.ExperienceReplayBuffer import SACExperienceReplayBuffer
from domain import ACTIONS
from environment.env import RenderSave, RenderSaveExtended
from loss_logger import LossLogger, ALPHA_VALUES
from tensorflow_probability import distributions as tfp
from scipy.special import kl_div

from timer import MultiTimer


class PPOAgent:

    #todo
    def __init__(self, environment_batcher, loss_logger: LossLogger, replay_buffer: SACExperienceReplayBuffer, self_play: bool, agent_ids: List[str], action_dim,
                 actor_network_generator, critic_network_generator, recurrent: bool, epsilon: float,
                 learning_rate: float, gamma: float, tau: float, reward_scale: float, alpha: float, l_alpha: float, social_reward_weight:float,
                 batch_size: int, model_path: str, target_entropy: float, seed: int, gae_lamda: float, kld_threshold: float, social_influence_sample_size: int):
        self._environment_batcher = environment_batcher
        self._loss_logger = loss_logger
        self._self_play = self_play
        self._action_dim = action_dim
        self._social_reward_weight = social_reward_weight
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
        self._social_influence_sample_size = social_influence_sample_size
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


    @tf.function
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

    @tf.function
    def log_probs_and_entropy_from_policy(self, state, action):
        probability_groups = self._actor(state)
        probability_groups = probability_groups if type(probability_groups) == list else [probability_groups]
        return tf.concat([[log_probs[tf.cast(action,tf.bool)] for log_probs in probability_groups]],axis=-1)
        #distributions = [tfp.Categorical(probs=probability_group) for probability_group in probability_groups]
        #log_probs = [distribution.log_prob(tf.argmax(action,axis=1)) for distribution in distributions]
        #return log_probs

    @tf.function
    def _tf_train_step_actor(self,state,action,prob_old,advantage):
        with tf.GradientTape() as tape:
            probability_groups =  self._actor(state)
            if self._environment_batcher._stats.number_communication_channels > 0:
                probability_groups = tf.concat(probability_groups, axis=1)
            log_prob_current = tf.reduce_sum(probability_groups * action, axis=1) #tf.gather
            p = tf.math.exp(log_prob_current - prob_old)  # exp() to un do log(p)
            clipped_p = tf.clip_by_value(p, 1 - self._epsilon, 1 + self._epsilon)
            policy_loss = -tfm.reduce_mean(tfm.minimum(p * advantage, clipped_p * advantage))
            l_entropy_loss = -tf.reduce_mean(tf.reduce_sum(probability_groups[:,len(ACTIONS):]*tfm.exp(probability_groups[:,len(ACTIONS):]),axis=1))
            r_entropy_loss = -tf.reduce_mean(tf.reduce_sum(probability_groups[:,:len(ACTIONS)]*tfm.exp(probability_groups[:,:len(ACTIONS)]),axis=1))
            entropy_loss = self._alpha * r_entropy_loss + self._l_alpha * l_entropy_loss
            loss = policy_loss - entropy_loss
        gradients = tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(gradients, self._actor.trainable_variables))
        return log_prob_current, loss, l_entropy_loss, r_entropy_loss

    def train_step_actor(self, state, action, advantage, prob_old):
        if tf.reduce_sum(tf.cast(tf.math.is_nan(state),tf.float32)) > 0:
                raise Exception(f"Actor log_prob_current contain nan, {state} {tf.reduce_sum(tf.cast(tf.math.is_nan(state),tf.float32))}")
        if self._self_play:
            with tf.GradientTape() as tape:
                #log_prob_current = self.log_probs_and_entropy_from_policy(state, action)
                log_prob_current,loss,l_entropy_loss, r_entropy_loss = self._tf_train_step_actor(state,action,prob_old,advantage=advantage)
            # check whether actor weights contain nan
            if tf.reduce_sum(tf.cast(tf.math.is_nan(state),tf.float32)) > 0:
                raise Exception(f"Actor log_prob_current contain nan, {state} {tf.reduce_sum(tf.cast(tf.math.is_nan(state),tf.float32))}")
            log_ratio = prob_old - log_prob_current #todo check if this is correct
            kld = tf.math.reduce_mean((tf.math.exp(log_ratio) - 1) - log_ratio)
            early_stopping = kld > self._kld_threshold
            return loss, early_stopping, r_entropy_loss, l_entropy_loss, kld
        else:
            raise NotImplementedError()


    @tf.function
    def sample_actions(self, states, actor, generator_index:int = 0, deterministic=False):
        log_prob_groups = actor(states) # first one for actions, rest for communication channels
        log_prob_groups = log_prob_groups if type(log_prob_groups) == list else [log_prob_groups]
        if deterministic:
            action_groups = [tf.argmax(probabilities, axis=1) for probabilities in log_prob_groups]
        else:
            #action_groups = [self._generators[generator_index].categorical(logits=log_probs, num_samples=1)[:,0] for log_probs in log_prob_groups]
            action_groups = [tf.random.stateless_categorical(logits=log_probs, num_samples=1, seed=self._generators[generator_index].make_seeds(2)[0])[:,0] for log_probs in log_prob_groups]
        if deterministic or self._environment_batcher._stats.number_communication_channels == 0:
            additional_com_samples_one_hot = None
        else:
            additional_com_samples = [tf.random.stateless_categorical(logits=log_probs, num_samples=self._social_influence_sample_size, seed=self._generators[generator_index].make_seeds(2)[0]) for log_probs in log_prob_groups[1:]]
            additional_com_samples_one_hot = None#tf.concat([tf.one_hot(sample, depth=self._environment_batcher._stats.size_vocabulary+1) for sample in additional_com_samples],axis=2)
        one_hot_action_groups = [tf.one_hot(actions, depth=probabilities.shape[-1]) for actions, probabilities in zip(action_groups,log_prob_groups)]
        # check whether probabilitygroups contains nan
        return tf.squeeze(tf.concat(one_hot_action_groups,axis=-1)), tf.concat(log_prob_groups,axis=-1), additional_com_samples_one_hot

    def random_additional_com_samples(self, stats):
        logits = tf.math.log(tf.ones(shape=(self._environment_batcher.size*self._social_influence_sample_size*stats.number_of_agents*stats.number_communication_channels, stats.size_vocabulary+1))/(stats.size_vocabulary+1))
        samples = tf.random.stateless_categorical(logits=logits, num_samples=1,
                                                  seed=self._generators[0].make_seeds(2)[0])
        one_hot_samples = tf.one_hot(samples, depth=stats.size_vocabulary + 1)
        reshaped_samples = tf.reshape(one_hot_samples, shape=(
            self._environment_batcher.size, stats.number_of_agents, self._social_influence_sample_size,
            stats.number_communication_channels * (stats.size_vocabulary + 1)))
        return reshaped_samples

    def sample_additional_probs(self, states, actor, additional_com_samples, original_action, multitimer: MultiTimer):
        #multitimer.start("repeat_states")
        repeated_states = np.array(tf.repeat(tf.expand_dims(states, axis=1), axis=1, repeats=additional_com_samples.shape[2]))
        #multitimer.stop("repeat_states")
        mean_prob_original_action = np.zeros(shape=(self._environment_batcher.size, len(self._agent_ids), len(self._agent_ids)-1, len(ACTIONS)))
        for index_influencer in range(states.shape[-2]): #for each agent
            for index_listener in range(states.shape[-2]): #for all other agents
                if index_listener == index_influencer:
                    continue
                #multitimer.start("pos_of_com")
                pos_of_com = self._environment_batcher._stats.index_of_communication_in_observation(agent_index=index_listener, speaker_index=index_influencer)
                #multitimer.stop("pos_of_com")
                #multitimer.start("repeated_states2")
                repeated_states[:,:,-1,index_listener,pos_of_com:pos_of_com+additional_com_samples.shape[3]] = additional_com_samples[:,index_influencer]
                #multitimer.stop("repeated_states2")
                #multitimer.start("relevant_states")
                relevant_states = np.reshape(repeated_states[:, :, :, index_listener, :], newshape=(
                    repeated_states.shape[0] * repeated_states.shape[1], repeated_states.shape[2],
                    repeated_states.shape[4]))
                #multitimer.stop("relevant_states")
                #multitimer.start("actor")
                log_prob_actions = actor(relevant_states)[0]
                #multitimer.stop("actor")
                #multitimer.start("reshape_mean_set")
                reshaped_log_prob_actions = tf.reshape(log_prob_actions, shape=(repeated_states.shape[0], repeated_states.shape[1], len(ACTIONS)))
                means_log_probs = tf.reduce_mean(reshaped_log_prob_actions,axis=1)
                mean_prob_original_action[:,index_influencer,index_listener if index_listener<index_influencer else index_listener-1,:]  = means_log_probs
                #multitimer.stop("reshape_mean_set")
                #mean_prob_original_action[:,index_influencer,index_listener if index_listener<index_influencer else index_listener-1]  = tf.reduce_min(means_log_probs*relevant_actions,axis=1)
        return mean_prob_original_action


    def act_batched(self, batched_state, env_batcher,  deterministic:bool, include_social: bool, multitimer=None,) -> Tuple[Tuple, np.ndarray, np.ndarray]:
        if multitimer is not None:
            multitimer.start("batch_compute_action_dict")
        batched_actions, batched_action_probs, batched_additional_com_samples = self._wrap_batched_compute_actions_one_hot_and_log_prob(batched_state, deterministic)
        assert not tf.math.reduce_any(tf.math.is_nan(batched_action_probs)), f"{batched_action_probs}{batched_state}"
        if multitimer is not None:
            multitimer.stop("batch_compute_action_dict")
            multitimer.start("env_step")
        observation_prime, reward, done, truncated = env_batcher.step(batched_actions)
        if multitimer is not None:
            multitimer.stop("env_step")
            multitimer.start("sample_additional_probs")
        if include_social:
            #bb = tf.repeat(tf.expand_dims(batched_actions[:, :, 5:], axis=2), axis=2, repeats=30)
            #batched_additional_com_samples = tf.stack([bb[:, 1], bb[:, 0]], axis=1)
            actual_probs = np.moveaxis(np.array([self._actor(observation_prime[:,:,agent_index,:])[0] for agent_index in range(len(self._agent_ids))]),0,1)
            batched_additional_com_samples = self.random_additional_com_samples(env_batcher._stats)
            mean_prob_alternatives = self.sample_additional_probs(observation_prime, self._actor, batched_additional_com_samples, batched_actions, multitimer=multitimer)
            social_influence_pairs = np.array([[[(actual_probs[index,index_listener,:len(ACTIONS)], mean_prob_alternatives[index,index_influencer,index_listener if index_listener<index_influencer else index_listener -1]) for index_listener in range(len(self._agent_ids))if index_listener!=index_influencer]   for index_influencer in range(len(self._agent_ids)) ] for index in range(self._environment_batcher.size)])
            exp_social_influence_pairs = np.exp(social_influence_pairs)
            kl_div_ = kl_div(exp_social_influence_pairs[:, :, :, 0], exp_social_influence_pairs[:, :, :, 1])
            kl_div_normalized = np.log(kl_div_ + 1)
            kl_div_by_agent = np.sum(kl_div_normalized, axis=2)
            social_reward = np.sum(kl_div_by_agent, axis=2)
        else:
            social_reward = 0.
        reward += self._social_reward_weight * social_reward
        assert not tf.math.reduce_any(tf.math.is_nan(observation_prime)), f"{observation_prime}{batched_state}"
        if multitimer is not None:
            multitimer.stop("sample_additional_probs")
        return (
            (batched_actions, observation_prime, reward, tf.math.logical_or(done, truncated)),batched_action_probs, social_reward)
    def act(self, state, env,  deterministic:bool, multitimer=None) -> Tuple[Tuple, np.ndarray]:
        actions, log_probs, _ = self._compute_actions_one_hot_and_log_prob(state, deterministic)

        observation_prime, reward, done, truncated = env.step(actions)
        return (
            (actions, observation_prime, reward, done or truncated),log_probs)


    def _wrap_batched_compute_actions_one_hot_and_log_prob(self, state, deterministic):
        actions_one_hot, log_probs, additional_com_samples = self._batched_compute_actions_one_hot_and_log_prob(state, deterministic)
        if state.shape[0]==1:
            actions_one_hot, log_probs = tf.expand_dims(actions_one_hot, axis=1), log_probs
        return np.moveaxis(np.array(actions_one_hot), 0, 1), np.moveaxis(np.array(log_probs), 0, 1), additional_com_samples#np.moveaxis(np.array(additional_com_samples), 0, 1)


    @tf.function
    def _batched_compute_actions_one_hot_and_log_prob(self, state, deterministic):
        actions_one_hot, log_probs, additional_com_samples = list(zip(*[self.sample_actions(deterministic=deterministic,generator_index=index,
                                                                           states= tf.convert_to_tensor(state[:,:,index,:]),
                                                                           actor=self._actor) for index in range(len(self._agent_ids))]))
        return actions_one_hot, log_probs, additional_com_samples


    def _compute_actions_one_hot_and_log_prob(self, state, deterministic):
        assert state.shape == (self._environment_batcher._stats.recurrency, len(self._agent_ids), self._environment_batcher._stats.observation_dimension)
        actions_one_hot, log_probs, additional_com_samples = list(zip(*[self.sample_actions(deterministic=deterministic,generator_index=index,
                                                                           states=tf.expand_dims(
                                                                               tf.convert_to_tensor(state[:,index,:]), axis=0),
                                                                           actor=self._actor) for index in range(len(self._agent_ids))]))
        return np.array(actions_one_hot), np.squeeze(np.array(log_probs)), additional_com_samples


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

    def sample_trajectories(self, steps_per_trajectory, render=False):
        observation = self._environment_batcher.reset_all()
        resets = 1
        rewards = []
        dones = []
        values = []
        observations = []
        actions = []
        probabilities = []
        current_render_list: List[RenderSaveExtended] = []
        render_list: List[List[RenderSaveExtended]] = []
        last_done = [False] * self._environment_batcher.size
        social_rewards = []

        with MultiTimer("env_step") as multitimer:
            for _ in range(steps_per_trajectory//self._environment_batcher.size):
                multitimer.start("act_batched")
                (action, new_observation, reward, next_done), log_probs, social_reward = self.act_batched(observation, self._environment_batcher, deterministic=False,multitimer=multitimer,include_social=self._environment_batcher._stats.number_communication_channels>0)
                multitimer.stop("act_batched")
                rewards.append(reward)
                social_rewards.append(social_reward)
                values.append(self.get_values(states=observation))
                dones.append(last_done)
                observations.append(observation)
                actions.append(action)
                probabilities.append(tf.reduce_sum(log_probs*action,axis=2))
                if render:
                    current_render_list.append((self._environment_batcher.render(index=0), np.exp(log_probs[0]),{id: (value, index) for id, value, index in zip(self._agent_ids,values[-1][0],range(100))}))
                multitimer.start("reset")
                observation = self._environment_batcher.reset(mask=next_done, observation_array=new_observation)
                multitimer.stop("reset")
                if render and next_done[0]:
                    current_render_list.append((self._environment_batcher.render(index=0), np.exp(log_probs[0]),{id: (value, index) for id, value, index in zip(self._agent_ids,values[-1][0],range(100))}))
                    render_list.append(current_render_list)
                    current_render_list = []
                last_done = next_done
        next_value = self.get_values(states=observation)
        np_rewards = np.array(rewards)
        np_values = np.array(values)
        np_dones = np.array(dones)
        multitimer.start("advantage")
        advantages_list = [self.estimate_advantage(rewards=np_rewards[:,index], values=np_values[:,index], dones=np_dones[:,index], next_value=next_value[index], next_done=next_done[index]) for index in range(len(next_done))]
        multitimer.stop("advantage")
        return (tf.convert_to_tensor([element for batch in observations for element in batch]), tf.convert_to_tensor([element for batch in actions for element in batch]),
                tf.convert_to_tensor([element for batch in zip(*advantages_list) for element in batch]),
                tf.convert_to_tensor([element for batch in rewards for element in batch]),
                tf.convert_to_tensor([element for batch in probabilities for element in batch])), render_list, np.average(social_rewards)