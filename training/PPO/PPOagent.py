from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from scipy.special import kl_div
from tensorflow import math as tfm
from typing_extensions import override

from domain import ACTIONS, RenderSaveExtended
from loss_logger import ACTOR_LOSS, KLD, COM_ENTROPY, ENTROPY, CRITIC_LOSS, V_VALUES
from timer import MultiTimer
from training.Agent import Agent, EarlyStopping
from training.ExperienceReplayBuffer import STATE_KEY, REWARD_KEY, ACTION_KEY
from training.PPO.ExperienceReplayBuffer import ADVANTAGE_KEY, PROB_OLD_KEY


class PPOAgent(Agent):


    #todo
    def __init__(self, environment_batcher, self_play: bool, agent_ids: List[str], actor_network_generator, critic_network_generator,
                 epsilon: float,
                 gamma: float, tau: float, alpha: float, com_alpha: float, social_reward_weight:float,
                 model_path: str, seed: int, gae_lamda: float, kld_threshold: float, social_influence_sample_size: int):
        super().__init__()
        self._environment_batcher_ = environment_batcher
        self._kld_threshold = kld_threshold
        self._gae_lamda = gae_lamda
        self._epsilon = epsilon
        self._init(self_play=self_play, agent_ids=agent_ids, actor_network_generator=actor_network_generator, actor_uses_log_probs=True,
                   critic_network_generator=critic_network_generator, gamma=gamma, tau=tau, mov_alpha=alpha, com_alpha=com_alpha,
                   model_path=model_path, seed=seed, social_influence_sample_size=social_influence_sample_size, social_reward_weight=social_reward_weight)


    @override
    @tf.function
    def train_step_critic(self, batch:Dict[str,tf.Tensor])->Dict[str,float]:
        states, returns = batch[STATE_KEY], batch[REWARD_KEY]
        reshaped_states = tf.reshape(states, shape=(states.shape[0], self._environment_batcher._stats.recurrency, self._environment_batcher._stats.observation_dimension * len(self._agent_ids)))
        losses = 0
        for critic in [self._critic_1, self._critic_2]:
            with tf.GradientTape() as tape:
                v_values = critic(reshaped_states)
                loss = self._mse(returns, v_values)
            gradients = tape.gradient(loss, critic.trainable_variables)
            critic.optimizer.apply_gradients(
                zip(gradients, critic.trainable_variables)
            )
            losses+=loss
        return {CRITIC_LOSS: losses, V_VALUES:tf.reduce_mean(v_values)}

    @tf.function
    @override
    def train_step_actor(self, batch: Dict[str, tf.Tensor])->Tuple[EarlyStopping, Dict[str,float]]:
        state, action, prob_old, advantage = batch[STATE_KEY], batch[ACTION_KEY], batch[PROB_OLD_KEY], batch[
            ADVANTAGE_KEY]
        with tf.GradientTape() as tape:
            probability_groups = self._actor(state)
            if self._environment_batcher._stats.number_communication_channels > 0:
                probability_groups = tf.concat(probability_groups, axis=1)
            log_prob_current = tf.reduce_sum(probability_groups * action, axis=1) #tf.gather
            p = tf.math.exp(log_prob_current - prob_old)  # exp() to un do log(p)
            clipped_p = tf.clip_by_value(p, 1 - self._epsilon, 1 + self._epsilon)
            policy_loss = -tfm.reduce_mean(tfm.minimum(p * advantage, clipped_p * advantage))
            com_entropy_loss = -tf.reduce_mean(tf.reduce_sum(probability_groups[:,len(ACTIONS):]*tfm.exp(probability_groups[:,len(ACTIONS):]),axis=1))
            mov_entropy_loss = -tf.reduce_mean(tf.reduce_sum(probability_groups[:,:len(ACTIONS)]*tfm.exp(probability_groups[:,:len(ACTIONS)]),axis=1))
            entropy_loss = self._mov_alpha * mov_entropy_loss + self._com_alpha * com_entropy_loss
            loss = policy_loss - entropy_loss
        gradients = tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(gradients, self._actor.trainable_variables))

        log_ratio = batch[PROB_OLD_KEY] - log_prob_current
        kld = tf.math.reduce_mean((tf.math.exp(log_ratio) - 1) - log_ratio)
        early_stopping = kld > self._kld_threshold
        return early_stopping, {ACTOR_LOSS: loss, ENTROPY: mov_entropy_loss, COM_ENTROPY: com_entropy_loss, KLD: kld}


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
        rewards, dones, values, observations, actions, probabilities, social_rewards = [], [], [], [], [], [], []
        current_render_list: List[RenderSaveExtended] = []
        render_list: List[List[RenderSaveExtended]] = []
        last_done = [False] * self._environment_batcher.size

        for _ in range(steps_per_trajectory//self._environment_batcher.size):
            (action, new_observation, reward, next_done), log_probs, social_reward = self.act_batched(observation, self._environment_batcher, deterministic=False,
                                                                                                          include_social=self._environment_batcher._stats.number_communication_channels > 0)
            rewards.append(reward)
            social_rewards.append(social_reward)
            values.append(self.get_values(states=observation))
            dones.append(last_done)
            observations.append(observation)
            actions.append(action)
            probabilities.append(tf.reduce_sum(log_probs*action,axis=2))
            if render:
                current_render_list.append((self._environment_batcher.render(index=0), np.exp(log_probs[0]),{id: (value, index) for id, value, index in zip(self._agent_ids,values[-1][0],range(100))}))
            observation = self._environment_batcher.reset(mask=next_done, observation_array=new_observation)
            if render and next_done[0]:
                current_render_list.append((self._environment_batcher.render(index=0), np.exp(log_probs[0]),{id: (value, index) for id, value, index in zip(self._agent_ids,values[-1][0],range(100))}))
                render_list.append(current_render_list)
                current_render_list = []
            last_done = next_done
        next_value = self.get_values(states=observation)
        np_rewards = np.array(rewards)
        np_values = np.array(values)
        np_dones = np.array(dones)
        advantages_list = [self.estimate_advantage(rewards=np_rewards[:,index], values=np_values[:,index], dones=np_dones[:,index], next_value=next_value[index], next_done=next_done[index]) for index in range(len(next_done))]
        arrays = {STATE_KEY: observations, ACTION_KEY: actions, ADVANTAGE_KEY: zip(*advantages_list), REWARD_KEY: rewards, PROB_OLD_KEY: probabilities}
        converted_arrays = {key: tf.convert_to_tensor([element for batch in array for element in batch]) for key, array in arrays.items()}
        return converted_arrays, render_list, social_rewards

    @property
    def _environment_batcher(self):
        return self._environment_batcher_
