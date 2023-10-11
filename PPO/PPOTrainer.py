from collections import defaultdict
from collections import defaultdict
from threading import Thread
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf

from PPO.ExperienceReplayBuffer import PPOExperienceReplayBuffer, ADVANTAGE_KEY, PROB_OLD_KEY
from PPO.PPOagent import PPOAgent
from SAC.ExperienceReplayBuffer import RecurrentExperienceReplayBuffer
from SAC.SACagent import SACAgent
from environment.env import RenderSaveExtended
from environment.envbatcher import EnvBatcher
from environment.render import render_permanently
from language.predictGoal import TrainPredictGoal
from loss_logger import LossLogger, CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, RETURNS, Q_VALUES, MAX_Q_VALUES, ALPHA_VALUES, \
    ENTROPY, L_ENTROPY, N_AGENT_RETURNS, KLD, AVG_ADVANTAGE, STD_ADVANTAGE, TEST_RETURNS, PREDICTED_GOAL, SOCIAL_REWARD
from domain import ACTIONS
from plots import plot_multiple
from timer import Timer, MultiTimer
from training.ExperienceReplayBuffer import ACTION_KEY, STATE_KEY, REWARD_KEY


class PPOTrainer:

    def __init__(self, environment, self_play: bool, agent_ids: List[str], state_dim, action_dim, from_save: bool,
                 actor_network_generator, critic_network_generator, max_replay_buffer_size: int, recurrent: bool, gae_lambda: float,social_influence_sample_size:int,social_reward_weight:float,
                 learning_rate, gamma, alpha,l_alpha:float, env_parallel: int,seed:int,run_name:str,epsilon:float,steps_per_trajectory:int,kld_threshold:float,
                 batch_size: int, target_entropy,model_path="model/", tau=0.005, reward_scale=1):
        self._loss_logger = LossLogger()
        self._env_batcher = EnvBatcher(env=environment, batch_size=env_parallel)
        self._replay_buffer = PPOExperienceReplayBuffer(state_dims=state_dim,action_dims= action_dim, agent_number=len(agent_ids),
                                                              max_size=batch_size*20, batch_size=batch_size, time_steps=environment.stats.recurrency)
        self._agent = PPOAgent(environment_batcher=self._env_batcher, loss_logger=self._loss_logger,seed=seed,
                               replay_buffer=self._replay_buffer, self_play=self_play, agent_ids=agent_ids,
                               action_dim=action_dim, actor_network_generator=actor_network_generator,
                               gae_lamda=gae_lambda,social_reward_weight=social_reward_weight,
                               critic_network_generator=critic_network_generator, recurrent=recurrent,
                               learning_rate=learning_rate, gamma=gamma, tau=tau, reward_scale=reward_scale,social_influence_sample_size=social_influence_sample_size,
                               alpha=alpha,l_alpha=l_alpha, batch_size=batch_size, model_path=model_path, target_entropy=target_entropy, epsilon=epsilon,kld_threshold=kld_threshold)
        self._environment = environment
        self._batch_size = batch_size
        self._agent_ids = agent_ids
        self._steps_per_trajectory = steps_per_trajectory
        self._run_name = run_name
        if from_save:
            self._agent.load_models(name="", run_name=run_name)
            self._loss_logger.load(path="logger")
        else:
            self._loss_logger.add_lists([CRITIC_LOSS, ACTOR_LOSS, Q_VALUES,  KLD, AVG_ADVANTAGE, STD_ADVANTAGE], smoothed=10)
            self._loss_logger.add_lists([ENTROPY, L_ENTROPY, TEST_RETURNS], smoothed=3)
            self._loss_logger.add_lists([RETURNS, SOCIAL_REWARD], smoothed=20)
            #self._loss_logger.add_lists([PREDICTED_GOAL], smoothed=1)


    def _extend_render_save(self, render_save: List[RenderSaveExtended], action_probs, observation_array: np.ndarray):
        render_save.append((self._env_batcher.render(index=0), action_probs,
                 self._agent._get_max_q_value(states=observation_array)))


    def train(self, epochs, environment_steps_before_training, pre_sampling_steps, render: bool, training_steps_per_update, run_desc: str=""):

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
        self._last_render_as_list = []
        if render:
            thread = Thread(target=render_permanently, args=[self._last_render_as_list])
            thread.start()
        print("start training!")
        for e in range(epochs):
            (obs, actions, advantages, rewards, probs_old), render_list, social_rewards = self._agent.sample_trajectories(steps_per_trajectory=self._steps_per_trajectory, render=render)
            if render:
                ...#self._last_render_as_list.extend(render_list)
            self._replay_buffer.add_transition_batch({STATE_KEY:obs, ACTION_KEY:actions, ADVANTAGE_KEY:advantages, PROB_OLD_KEY:probs_old, REWARD_KEY:rewards})
            for batch in list(self._replay_buffer.get_all_repeated(repetitions=4)):
                s_concat = tf.concat([batch[STATE_KEY][:,:,index,:] for index in range(self._environment.stats.number_of_agents)],axis=0)
                a_concat = tf.concat([batch[ACTION_KEY][:,index,:] for index in range(self._environment.stats.number_of_agents)],axis=0)
                probs_old_concat = tf.concat([batch[PROB_OLD_KEY][:,index] for index in range(self._environment.stats.number_of_agents)],axis=0)
                adv_concat = tf.concat([batch[ADVANTAGE_KEY][:, index] for index in range(self._environment.stats.number_of_agents)],
                                       axis=0)

                if tf.reduce_sum(tf.cast(tf.math.is_nan(s_concat), tf.float32)) > 0:
                    raise Exception(
                        f"Actor log_prob_current contain nan, {s_concat} {tf.reduce_sum(tf.cast(tf.math.is_nan(s_concat), tf.float32))}")
                actor_loss, early_stopping, entropy, l_entropy, kld = self._agent.train_step_actor(state=s_concat, action=a_concat, advantage=adv_concat, prob_old=probs_old_concat)
                self._loss_logger.add_aggregatable_values({ACTOR_LOSS: actor_loss, ENTROPY: entropy,L_ENTROPY:l_entropy, KLD: kld, SOCIAL_REWARD: social_rewards})
                if early_stopping:
                    break
                critic_loss, sample_q_value = self._agent.train_step_critic(states=batch[STATE_KEY], ret=batch[REWARD_KEY])
                self._loss_logger.add_aggregatable_values({CRITIC_LOSS: critic_loss, Q_VALUES: sample_q_value})
            self._loss_logger.add_value(identifier=RETURNS, value=tf.reduce_mean(rewards))
            self._loss_logger.add_value(identifier=AVG_ADVANTAGE, value=tf.reduce_mean(advantages))
            self._loss_logger.add_value(identifier=STD_ADVANTAGE, value=tf.math.reduce_std(advantages))
            self._loss_logger.avg_aggregatables([CRITIC_LOSS, ACTOR_LOSS, ENTROPY, L_ENTROPY, KLD, Q_VALUES, SOCIAL_REWARD])
            print("epoch", e, "avg reward {:.2f}".format(tf.reduce_mean(rewards)), "loss {:.2f}".format(actor_loss), "early_stopping",early_stopping)
            self._replay_buffer.clear()
            self._agent.update_target_weights()
            if e % 10 == 0:
                self._agent.save_models(name="", run_name=self._run_name)
                self._loss_logger.add_value(identifier=TEST_RETURNS,value=self.test(n_samples=10,render=render))
                self._loss_logger.save(path="logger")
                thread = Thread(target=plot_multiple, args=[self._run_name, self._loss_logger.all_smoothed()])
                thread.start()
            if self._environment.stats.number_communication_channels>0 and e%100 == 0:
                #self._loss_logger.add_value(identifier=PREDICTED_GOAL,value=TrainPredictGoal()(environment=self._environment, agent=self._agent))
                TrainPredictGoal()(environment=self._environment, agent=self._agent)
        print("training finished!")




    def test(self, n_samples: int, render: bool = False)->float:
        returns = []
        render_save: List[RenderSaveExtended] = []
        self._last_render_as_list.clear()
        for index in range(n_samples):
            observation_array, _ = self._environment.reset()
            action_probs = defaultdict(lambda: np.zeros(shape=(len(ACTIONS))))
            return_ = 0
            while True:
                if  render:
                    values = self._agent.get_values(tf.expand_dims(observation_array,axis=0))
                    values_by_agent = {agent: (values[0,index],0) for index, agent in enumerate(range(self._environment.stats.number_of_agents))}
                    render_save.append((self._environment.render(), action_probs, values_by_agent))
                (actions_array, new_observation_array, reward_array, done), action_probs = self._agent.act(
                    observation_array, deterministic=True, env=self._environment)
                return_ += sum(reward_array)
                if done:
                    returns.append(return_)
                    if render:
                        values = self._agent.get_values(tf.expand_dims(observation_array,axis=0))
                        values_by_agent = {agent: (values[0,index],0) for index, agent in enumerate(range(self._environment.stats.number_of_agents))}
                        render_save.append((self._environment.render(), action_probs, values_by_agent))
                        self._last_render_as_list.append(render_save)
                        render_save = []
                    print(
                        f"Finished test episode {index} with a return of {return_} after {self._environment.stats.time_step} time steps.")
                    break
                observation_array = new_observation_array
            # if index == 0:
            #    thread = Thread(target=render_episode,args=[render_save])
            #    thread.start()
        avg_return = np.mean(returns)
        print(f"The average return is {avg_return}")
        return avg_return

    @property
    def extended_shape(self):
        return (self._batch_size, len(self._agent_ids), self._environment.stats.recurrency, self._environment.stats.observation_dimension)
