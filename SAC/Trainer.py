from collections import defaultdict
from collections import defaultdict
from threading import Thread
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf

from SAC.ExperienceReplayBuffer import RecurrentExperienceReplayBuffer
from SAC.SACagent import SACAgent
from environment.env import RenderSaveExtended
from environment.envbatcher import EnvBatcher
from environment.render import render_permanently
from loss_logger import LossLogger, CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, RETURNS, Q_VALUES, MAX_Q_VALUES, ALPHA_VALUES, \
    ENTROPY, L_ENTROPY, N_AGENT_RETURNS
from domain import ACTIONS
from plots import plot_multiple
from timer import Timer, MultiTimer

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

    # todo cnn
    def __init__(self, environment, self_play: bool, agent_ids: List[str], state_dim, action_dim, from_save: bool,
                 actor_network_generator, critic_network_generator, max_replay_buffer_size: int, recurrent: bool,
                 learning_rate, gamma, alpha,l_alpha:float, env_parallel: int,seed:int,run_name:str,
                 batch_size: int, target_entropy,model_path="model/", tau=0.005, reward_scale=1):
        self._loss_logger = LossLogger()
        self._env_batcher = EnvBatcher(env=environment, batch_size=env_parallel)
        self._replay_buffer = RecurrentExperienceReplayBuffer(state_dims=state_dim,action_dims= action_dim, agent_number=len(agent_ids),
                                                              max_size=max_replay_buffer_size, batch_size=batch_size, time_steps=environment.stats.recurrency)
        self._agent = SACAgent(environment=environment, loss_logger=self._loss_logger,seed=seed,
                               replay_buffer=self._replay_buffer, self_play=self_play, agent_ids=agent_ids,
                               action_dim=action_dim, actor_network_generator=actor_network_generator,
                               critic_network_generator=critic_network_generator, recurrent=recurrent,
                               learning_rate=learning_rate, gamma=gamma, tau=tau, reward_scale=reward_scale,
                               alpha=alpha,l_alpha=l_alpha, batch_size=batch_size, model_path=model_path, target_entropy=target_entropy)
        self._environment = environment
        self._batch_size = batch_size
        self._agent_ids = agent_ids
        self._run_name = run_name
        if from_save:
            self._agent.load_models(name="", run_name=run_name)
            self._loss_logger.load(path="logger")
        else:
            self._loss_logger.add_lists([CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, Q_VALUES, MAX_Q_VALUES, ENTROPY, L_ENTROPY], smoothed=100)
            self._loss_logger.add_lists([ALPHA_VALUES], smoothed=10)
            self._loss_logger.add_lists([RETURNS, N_AGENT_RETURNS(2), N_AGENT_RETURNS(3)], smoothed=1000)

    def _env_step(self, observation_array, multitimer: Optional[MultiTimer], return_array: np.ndarray):
        (actions_array, new_observation_array, reward_array, done), action_probs = self._agent.act_batched(
            batched_state = observation_array, deterministic=False, env_batcher=self._env_batcher, multitimer=multitimer)
        self._replay_buffer.add_transition_batch(state=observation_array, action=actions_array,
                                           reward=reward_array,
                                           state_=new_observation_array, done=done)
        observation_array = new_observation_array
        return_array += np.sum(reward_array,axis=1)
        return observation_array, return_array, done, action_probs[0]

    def _extend_render_save(self, render_save: List[RenderSaveExtended], action_probs, observation_array: np.ndarray):
        render_save.append((self._env_batcher.render(index=0), action_probs,
                 self._agent._get_max_q_value(states=observation_array)))
    def _reset_env(self, epoch_array: List[int], steps_total: int, render_save: List[RenderSaveExtended], render: bool, return_array: np.ndarray, done_mask: List[bool], observation_array: np.ndarray):
            if done_mask[0]:
                print("epoch:", epoch_array[0], "steps:", steps_total,
                      "actor-loss: {:.2f}".format(self._loss_logger.last(ACTOR_LOSS)),
                      "critic-loss: {:.2f}".format(self._loss_logger.last(CRITIC_LOSS)),
                      "return: {:.2f}".format(self._loss_logger.last(RETURNS)),
                      "avg return: {:.2f}".format(self._loss_logger.avg_last(RETURNS, 40)))
                if render:
                    self._last_render_as_list.append(render_save)
                    render_save = []
                    if len(self._last_render_as_list) > 1:
                        self._last_render_as_list.pop(0)
            self._loss_logger.add_value_list(identifier=RETURNS, values=return_array[done_mask])
            self._loss_logger.add_value_list(values=return_array[tf.math.logical_and(done_mask,(self._env_batcher.env_types == 2))], identifier=N_AGENT_RETURNS(2))
            self._loss_logger.add_value_list(values =return_array[tf.math.logical_and(done_mask,(self._env_batcher.env_types == 3))], identifier=N_AGENT_RETURNS(3))
            #self._loss_logger.add_value(identifier=self._environment.current_type, value=ret)
            return_array[done_mask] = 0
            observation_array = self._env_batcher.reset(observation_array=observation_array, mask=done_mask)
            epoch_array[done_mask] += self._env_batcher.size
            if render and done_mask[0]:
                self._extend_render_save(render_save=render_save, action_probs=np.zeros(shape=(len(self._agent_ids),self._environment.stats.action_dimension)), observation_array=observation_array[0])
            if epoch_array[0] % (self._env_batcher.size*100) == 0 and done_mask[0]:
                self.test(n_samples=20, verbose_samples=0)
                self._agent.save_models(name="", run_name=self._run_name)
                self._loss_logger.save(path="logger")
                thread = Thread(target=plot_multiple, args=[self._run_name, self._loss_logger.all_smoothed()])
                thread.start()
            return observation_array, epoch_array, return_array

    def train(self, epochs, environment_steps_before_training, pre_sampling_steps, render: bool, training_steps_per_update, run_desc: str=""):
        # loggen aber richtig
        # timetracing

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
        print("noc",self._environment.stats.number_communication_channels)
        self._pre_sample(pre_sampling_steps=pre_sampling_steps)
        if render:
            self._last_render_as_list = []
            thread = Thread(target=render_permanently, args=[self._last_render_as_list])
            thread.start()
        print("start training!")

        observation_array = self._env_batcher.reset_all()
        # default values for first iteration:
        done = np.array([False]*self._env_batcher.size)
        action_probs_0: Dict[str, np.ndarray] = np.zeros(shape=(len(self._agent_ids),self._environment.stats.action_dimension))
        return_array = np.zeros(shape=(self._env_batcher.size))

        render_save = []
        epoch_array = np.array(list(range(self._env_batcher.size)))
        steps_total = 0
        while True:
            steps = 0
            while steps < environment_steps_before_training:
                    if render:
                        self._extend_render_save(render_save=render_save, action_probs=action_probs_0, observation_array=observation_array[0])
                    observation_array, epoch_array, ret =self._reset_env(epoch_array=epoch_array, steps_total=steps_total, render_save=render_save, render=render, return_array=return_array, done_mask=done, observation_array=observation_array)
                    if min(epoch_array) >= epochs:
                        print("training finished!")
                        return
                    observation_array, ret, done, action_probs_0 = self._env_step(
                        observation_array=observation_array, multitimer=None, return_array=return_array
                    )
                    steps += self._env_batcher.size
                    steps_total += self._env_batcher.size



            for _ in range(training_steps_per_update):
                    self.learn()


    def learn(self) -> None:
            states, actions, rewards, states_prime, dones = self._replay_buffer.sample_batch()
            actor_loss, q_values, max_q_values = self._agent.train_step_actor(states)
            critic_loss, abs_1, abs_2, log_probs, entropy, l_entropy = self._agent.train_step_critic(
                states=tf.reshape(tensor=states, shape=self.extended_shape),
                actions=tf.reshape(tensor=actions, shape=(
                    self._batch_size, len(self._agent_ids) * self._environment.stats.action_dimension)),
                rewards=tf.reshape(tensor=rewards, shape=(self._batch_size, len(self._agent_ids))),
                states_prime=tf.reshape(tensor=states_prime, shape=self.extended_shape),
                dones=tf.reshape(tensor=dones, shape=(self._batch_size)),
            )
            #self._agent.train_step_temperature(states)
            self._agent.update_target_weights()
            self._loss_logger.add_aggregatable_values(
                {CRITIC_LOSS: critic_loss, LOG_PROBS: log_probs, ACTOR_LOSS: actor_loss, Q_VALUES: q_values,
                 MAX_Q_VALUES: max_q_values, ENTROPY: entropy,L_ENTROPY: l_entropy})
            self._loss_logger.avg_aggregatables([CRITIC_LOSS, LOG_PROBS, ACTOR_LOSS, Q_VALUES, MAX_Q_VALUES, ENTROPY, L_ENTROPY])

    # todo sachen auf notion packen

    def _pre_sample(self, pre_sampling_steps: int):
        print(f"Random exploration for {pre_sampling_steps} steps!")
        observation_array, _ = self._environment.reset()
        ret = 0
        for _ in range(pre_sampling_steps):
            (actions_array, new_observation_array, reward_array, done), _ = self._agent.act(state=observation_array,
                                                                                              deterministic=False, env=self._environment)
            ret += sum(reward_array)
            self._replay_buffer.add_transition(state=observation_array, action=actions_array, reward=reward_array,
                                               state_=new_observation_array, done=done)
            if done:
                ret = 0
                observation_array, _ = self._environment.reset()
            else:
                observation_array = new_observation_array

    def test(self, n_samples: int, verbose_samples: int, render: bool = False):
        returns = []
        render_save = []
        for index in range(n_samples):
            observation_array,_ = self._environment.reset()
            action_probs = defaultdict(lambda: np.zeros(shape=(1, len(ACTIONS))))
            return_ = 0
            while True:
                if index < verbose_samples and render:
                    render_save.append((self._environment.render(), action_probs))
                (actions_array, new_observation_array, reward_array, done), action_probs = self._agent.act(
                    observation_array, deterministic=True, env=self._environment)
                return_ += sum(reward_array)
                if done:
                    returns.append(return_)
                    if index < verbose_samples and render:
                        render_save.append(self._environment.render())
                    print(
                        f"Finished test episode {index} with a return of {return_} after {self._environment.stats.time_step} time steps.")
                    break
                observation_array = new_observation_array
            # if index == 0:
            #    thread = Thread(target=render_episode,args=[render_save])
            #    thread.start()
        print(f"The average return is {np.mean(returns)}")

    @property
    def extended_shape(self):
        return (self._batch_size, len(self._agent_ids), self._environment.stats.recurrency, self._environment.stats.observation_dimension)
