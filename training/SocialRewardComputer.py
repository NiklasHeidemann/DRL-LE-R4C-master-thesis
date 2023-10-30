import numpy as np
import tensorflow as tf
from scipy.special import kl_div

from domain import ACTIONS
from environment.stats import Stats


class SocialRewardComputer:

    def __init__(self, stats: Stats, social_influence_sample_size: int, num_agents: int, generators: tf.random.Generator,):
        self._stats = stats
        self._social_influence_sample_size = social_influence_sample_size
        self._num_agents = num_agents
        self._mse = tf.keras.losses.MeanSquaredError()
        self._generators = generators
    def __call__(self, observation_prime, deterministic: bool, actor: tf.keras.Model)->np.ndarray:
        if self._stats.number_communication_channels == 0:
            return np.zeros(shape=(self._num_agents,))
        actual_probs = np.moveaxis(np.array(
            [actor(observation_prime[:, :, agent_index, :])[0] for agent_index in
             range(self._num_agents)]), 0, 1)
        if deterministic:
            actual_probs = tf.one_hot(tf.argmax(actual_probs, axis=2), depth=actual_probs.shape[2])
        batched_additional_com_samples = self.random_additional_com_samples(stats=self._stats,
                                                                            size=observation_prime.shape[0])
        mean_prob_alternatives = self.sample_additional_probs(observation_prime, batched_additional_com_samples,
                                                              deterministic=deterministic, actor=actor)
        social_influence_pairs = np.array([[[(actual_probs[index, index_listener, :len(ACTIONS)],
                                              mean_prob_alternatives[
                                                  index, index_influencer, index_listener if index_listener < index_influencer else index_listener - 1])
                                             for index_listener in range(self._num_agents) if
                                             index_listener != index_influencer] for index_influencer in
                                            range(self._num_agents)] for index in
                                           range(observation_prime.shape[0])])
        if deterministic:
            social_reward = [self._mse(social_influence_pairs[:, agent_index, :, 0],
                                       social_influence_pairs[:, agent_index, :, 1]) for agent_index in
                             range(self._num_agents)]
        else:
            exp_social_influence_pairs = np.exp(social_influence_pairs)
            kl_div_ = kl_div(exp_social_influence_pairs[:, :, :, 0], exp_social_influence_pairs[:, :, :, 1])
            kl_div_normalized = np.log(kl_div_ + 1)
            kl_div_by_agent = np.sum(kl_div_normalized, axis=2)
            social_reward = np.sum(kl_div_by_agent, axis=2)
        return social_reward

    def sample_additional_probs(self, states, additional_com_samples, deterministic: bool, actor: tf.keras.Model):
        repeated_states = np.array(
            tf.repeat(tf.expand_dims(states, axis=1), axis=1, repeats=additional_com_samples.shape[2]))
        mean_prob_original_action = np.zeros(
            shape=(states.shape[0], self._num_agents, self._num_agents - 1, len(ACTIONS)))
        for index_influencer in range(states.shape[-2]):  # for each agent
            for index_listener in range(states.shape[-2]):  # for all other agents
                if index_listener == index_influencer:
                    continue
                pos_of_com = self._stats.index_of_communication_in_observation(
                    agent_index=index_listener, speaker_index=index_influencer)
                repeated_states[:, :, -1, index_listener,
                pos_of_com:pos_of_com + additional_com_samples.shape[3]] = additional_com_samples[:, index_influencer]
                relevant_states = np.reshape(repeated_states[:, :, :, index_listener, :], newshape=(
                    repeated_states.shape[0] * repeated_states.shape[1], repeated_states.shape[2],
                    repeated_states.shape[4]))
                log_prob_actions = actor(relevant_states)[0]
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
