from collections import defaultdict
import datetime
from collections import defaultdict
from threading import Thread
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd

from SAC.ExperienceReplayBuffer import RecurrentExperienceReplayBuffer
from SAC.SACagent import SACAgent
from environment.render import render_permanently
from loss_logger import LossLogger, CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, RETURNS, Q_VALUES, OTHER_ACTOR_LOSS, \
    MAX_Q_VALUES, ALPHA_VALUES, ENTROPY
from params import BATCH_SIZE, LEARNING_RATE, TIME_STEPS, TRAININGS_PER_TRAINING, ALPHA, ACTIONS, GAMMA, TARGET_ENTROPY
from plots import plot_multiple


class Trainer:
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
    def __init__(self, environment, self_play: bool, agent_ids: List[str], state_dim, action_dim, from_save: bool,
                 actor_network_generator, critic_network_generator, max_replay_buffer_size: int, recurrent: bool,
                 learning_rate=LEARNING_RATE, gamma=GAMMA, tau=0.005, reward_scale=1, alpha=ALPHA,
                 batch_size: int = BATCH_SIZE, model_path="model/", target_entropy=TARGET_ENTROPY):
        self._loss_logger = LossLogger()
        self._replay_buffer = RecurrentExperienceReplayBuffer(state_dim, action_dim, agent_number=len(agent_ids),
                                                             max_size=max_replay_buffer_size, batch_size=batch_size)
        self._agent = SACAgent(environment=environment, loss_logger=self._loss_logger, replay_buffer=self._replay_buffer, self_play=self_play, agent_ids=agent_ids, action_dim=action_dim,actor_network_generator=actor_network_generator,
                               critic_network_generator=critic_network_generator, recurrent=recurrent, learning_rate=learning_rate, gamma=gamma, tau=tau, reward_scale=reward_scale, alpha=alpha,batch_size=batch_size,model_path=model_path, target_entropy=target_entropy)
        self._environment = environment
        self._batch_size = batch_size
        self._agent_ids = agent_ids
        if from_save:
            self._agent.load_models(name="")

    def train(self, epochs, environment_steps_before_training=1, training_steps_per_update=1,
              max_environment_steps_per_epoch=None, pre_sampling_steps=1024):
        """
        trains the SAC Agent
        :param epochs: Number of epochs to train.
            One epoch is finished if the agents are done ore the maximum steps are reached
        :param environment_steps_before_training=1: Number of steps the agent takes in the environment
            before the training cycle starts (the networks are updated after each step in the environment)
        :param training_steps_per_update=1: Number of times the networks are update per update cykle
        :param max_environment_steps_per_epoch=None: Maximal number of staps taken in the environment in an epoch
        :param pre_sampling_steps=1024: Number of exploration steps sampled to the replay buffer before training starts
        """
        self._pre_sample(pre_sampling_steps=pre_sampling_steps)
        print("start training!")
        self._loss_logger.add_lists([CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, Q_VALUES, MAX_Q_VALUES], smoothed=100)
        self._loss_logger.add_lists([ALPHA_VALUES, ENTROPY], smoothed=10)
        self._loss_logger.add_lists([RETURNS], smoothed=1000)
        return_dict = defaultdict(float)
        observation_dict = self._environment.reset()
        action_probs = defaultdict(lambda: np.zeros(shape=(1, len(ACTIONS))))
        done_dict = {"": False}
        ret = 0
        epoch = 0
        steps = 0
        j = 0
        render_save = []
        last_render_as_list = []
        thread = Thread(target=render_permanently, args=[last_render_as_list])
        thread.start()
        while True:
            i = 0
            while i < environment_steps_before_training or self._replay_buffer.size() < self._batch_size:
                if epoch % 10 == 0:
                    render_save.append(
                        (self._environment.render(), action_probs, self._agent._get_max_q_value(states=observation_dict)))
                if False not in done_dict.values() or (
                        max_environment_steps_per_epoch is not None and j >= max_environment_steps_per_epoch):
                    if epoch % 10 == 0:
                        print("epoch:", epoch, "steps:", steps,
                              "actor-loss: {:.2f}".format(self._loss_logger.last(ACTOR_LOSS)),
                              "critic-loss: {:.2f}".format(self._loss_logger.last(CRITIC_LOSS)),
                              "return: {:.2f}".format(self._loss_logger.last(RETURNS)),
                              "avg return: {:.2f}".format(self._loss_logger.avg_last(RETURNS, 10)))
                        last_render_as_list.append(render_save)
                        render_save = []
                        if len(last_render_as_list) > 1:
                            last_render_as_list.pop(0)
                    observation_dict = self._environment.reset()
                    self._loss_logger.add_value(identifier=RETURNS, value=ret)
                    ret = 0
                    return_dict = defaultdict(float)
                    epoch += 1
                    if epoch % 1000 == 0:
                        self.test(n_samples=20, verbose_samples=0)
                    if epoch % 1000 == 0 or epoch >= epochs:
                        self._agent.save_models(name="")
                    if epoch % 1000 == 0 or epoch >= epochs:
                        thread = Thread(target=plot_multiple, args=[self._loss_logger.all_smoothed()])
                        thread.start()
                    if epoch >= epochs:
                        print("training finished!")
                        return
                (actions_dict, new_observation_dict, reward_dict, done_dict), action_probs = self._agent.act(
                    observation_dict,deterministic=False)
                self._replay_buffer.add_transition(state=observation_dict, action=actions_dict, reward=reward_dict,
                                                  state_=new_observation_dict, done=done_dict)
                observation_dict = new_observation_dict
                steps += 1
                ret += sum(reward_dict.values())
                return_dict = {key: return_dict[key] + reward_dict[key] for key in reward_dict.keys()}
                i += 1
                j += 1
            for _ in range(training_steps_per_update):
                self.learn()

    def learn(self) -> None:
        for _ in range(TRAININGS_PER_TRAINING):
            states, actions, rewards, states_prime, dones = self._replay_buffer.sample_batch()
            actor_loss, q_values, max_q_values = self._agent.train_step_actor(states)
            critic_loss, log_probs, entropy = self._agent.train_step_critic(
                states=tf.reshape(tensor=states, shape=self.extended_shape),
                actions=tf.reshape(tensor=actions, shape=(
                    self._batch_size, len(self._agent_ids) * self._environment.stats.action_dimension)),
                rewards=tf.reshape(tensor=rewards, shape=(self._batch_size, len(self._agent_ids))),
                states_prime=tf.reshape(tensor=states_prime, shape=self.extended_shape),
                dones=tf.reshape(tensor=dones, shape=(self._batch_size, len(self._agent_ids))),
            )
            self._agent.train_step_temperature(states)
            self._agent.update_target_weights()
            self._loss_logger.add_aggregatable_values(
                {CRITIC_LOSS: critic_loss, LOG_PROBS: log_probs, ACTOR_LOSS: actor_loss, Q_VALUES: q_values,
                 MAX_Q_VALUES: max_q_values, ENTROPY: entropy})
        self._loss_logger.avg_aggregatables([CRITIC_LOSS, LOG_PROBS, ACTOR_LOSS, Q_VALUES, MAX_Q_VALUES, ENTROPY])

    # todo free-for-all szenarien
    # todo relatedwork section skizzieren
    # todo sachen auf notion packen

    def _pre_sample(self, pre_sampling_steps: int):
        print(f"Random exploration for {pre_sampling_steps} steps!")
        observation_dict = self._environment.reset()
        ret = 0
        for _ in range(pre_sampling_steps):
            (actions_dict, new_observation_dict, reward_dict, done_dict), _ = self._agent.act(observation_dict, deterministic=False)
            ret += sum(reward_dict.values())
            self._replay_buffer.add_transition(state=observation_dict, action=actions_dict, reward=reward_dict,
                                              state_=new_observation_dict, done=done_dict)
            if False not in done_dict.values():
                ret = 0
                observation_dict = self._environment.reset()
            else:
                observation_dict = new_observation_dict

    def test(self, n_samples: int, verbose_samples: int):
        returns = []
        render_save = []
        for index in range(n_samples):
            observation_dict = self._environment.reset()
            action_probs = defaultdict(lambda: np.zeros(shape=(1, len(ACTIONS))))
            return_ = 0
            while True:
                if index < verbose_samples:
                    render_save.append((self._environment.render(), action_probs))
                (actions_dict, new_observation_dict, reward_dict, done_dict), action_probs = self._agent.act(
                    observation_dict,deterministic=True)
                if index < verbose_samples:
                    print(actions_dict)
                return_ += sum(reward_dict.values())
                if False not in done_dict.values():
                    returns.append(return_)
                    if index < verbose_samples:
                        render_save.append(self._environment.render())
                    print(
                        f"Finished test episode {index} with a return of {return_} after {self._environment.stats.time_step} time steps.")
                    break
                observation_dict = new_observation_dict
            # if index == 0:
            #    thread = Thread(target=render_episode,args=[render_save])
            #    thread.start()
        print(f"The average return is {np.mean(returns)}")

    @property
    def extended_shape(self):
        return (self._batch_size, len(self._agent_ids), TIME_STEPS, self._environment.stats.observation_dimension)

