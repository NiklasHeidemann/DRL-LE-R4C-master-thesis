from threading import Thread
from typing import List, Optional

import numpy as np
import tensorflow as tf
from typing_extensions import override

from environment.envbatcher import EnvBatcher
from environment.render import render_permanently
from utils.loss_logger import CRITIC_LOSS, ACTOR_LOSS, RETURNS, V_VALUES, ENTROPY, COM_ENTROPY, KLD, AVG_ADVANTAGE, \
    STD_ADVANTAGE, TEST_RETURNS, \
    SOCIAL_REWARD, PREDICTED_GOAL, TEST_SOCIAL_RETURNS
from training.ExperienceReplayBuffer import STATE_KEY, REWARD_KEY
from training.PPO.ExperienceReplayBuffer import PPOExperienceReplayBuffer, ADVANTAGE_KEY
from training.PPO.PPOagent import PPOAgent
from training.Trainer import Trainer


class PPOTrainer(Trainer):

    def __init__(self, environment, agent_ids: List[str], state_dim, action_dim, from_save: bool,
                 actor_network_generator, critic_network_generator, gae_lambda: float, epsilon:Optional[float],
                 social_influence_sample_size: int, social_reward_weight: float, tau: float,
                 gamma: float, alpha: float, com_alpha: float, env_parallel: int, seed: int, run_name: str,
                 ppo_epsilon: float, steps_per_trajectory: int, kld_threshold: float,
                 batch_size: int, model_path="model/"):
        self._env_batcher = EnvBatcher(env=environment, batch_size=env_parallel)
        replay_buffer = PPOExperienceReplayBuffer(state_dims=state_dim, action_dims=action_dim,
                                                        agent_number=len(agent_ids),
                                                        max_size=batch_size * 20, batch_size=batch_size,
                                                        time_steps=environment.stats.recurrency)
        agent = PPOAgent(environment_batcher=self._env_batcher, seed=seed,
                          agent_ids=agent_ids,
                         actor_network_generator=actor_network_generator,
                         gae_lamda=gae_lambda, social_reward_weight=social_reward_weight,
                         critic_network_generator=critic_network_generator, gamma=gamma, tau=tau,
                         social_influence_sample_size=social_influence_sample_size,
                         alpha=alpha, com_alpha=com_alpha, model_path=model_path,
                         epsilon=epsilon, kld_threshold=kld_threshold, ppo_epsilon=ppo_epsilon)
        metrics =        {
            10: [CRITIC_LOSS, ACTOR_LOSS, V_VALUES, KLD, AVG_ADVANTAGE, STD_ADVANTAGE],
        3: [ENTROPY, COM_ENTROPY, TEST_RETURNS, TEST_SOCIAL_RETURNS],
            20: [RETURNS, SOCIAL_REWARD],
            1: [PREDICTED_GOAL],
        }
        self._init(environment=environment, agent=agent, replay_buffer=replay_buffer, run_name=run_name,
                   from_save=from_save,metrics=metrics)
        self._steps_per_trajectory = steps_per_trajectory

    @override
    def train(self, num_epochs: int, render: bool):

        """
        trains the SAC Agent
        :param num_epochs: Number of epochs to train.
        :param render: If true, the environment is rendered during testing.
        """
        self._last_render_as_list = []
        if render:
            thread = Thread(target=render_permanently, args=[self._last_render_as_list])
            thread.start()
        print("start training!")
        epoch_this_training = 0
        while self.epoch < num_epochs:
            arrays, render_list, social_rewards = self._agent.sample_trajectories(
                steps_per_trajectory=self._steps_per_trajectory, render=render)
            self._replay_buffer.add_transition_batch(arrays)
            self._loss_logger.add_values({SOCIAL_REWARD:np.average(social_rewards),
                                          RETURNS: tf.reduce_mean(arrays[REWARD_KEY]),
                                            AVG_ADVANTAGE: tf.reduce_mean(arrays[ADVANTAGE_KEY]),
                                            STD_ADVANTAGE: tf.math.reduce_std(arrays[ADVANTAGE_KEY])})
            actor_metrics, critic_metrics = {},{}
            for batch in list(self._replay_buffer.get_all_repeated(repetitions=4)):
                batch_concat = {
                    key: tf.concat([array[:, :, index, :] if key==STATE_KEY else array[:,index] for index in range(self._environment.stats.number_of_agents)],axis=0) for key, array in batch.items()
                }
                early_stopping, actor_metrics= self._agent.train_step_actor(batch=batch_concat)
                self._loss_logger.add_aggregatable_values(actor_metrics)
                if early_stopping:
                    break
                critic_metrics = self._agent.train_step_critic(batch=batch)
                self._loss_logger.add_aggregatable_values(critic_metrics)
            self._loss_logger.avg_aggregatables(
                list(actor_metrics.keys())+list(critic_metrics.keys()))
            print(f"{self._run_name}: epoch (now/all_time/max): ({epoch_this_training}/{self.epoch}/{num_epochs})", "avg reward {:.2f}".format(tf.reduce_mean(arrays[REWARD_KEY])), "early_stopping", early_stopping)
            self._replay_buffer.clear()
            self._agent.update_target_weights()
            if self.epoch % 10 == 0:
                self._every_few_epochs(render=render)
            if self._environment.stats.number_communication_channels > 0 and self.epoch % 100 == 0:
                self._loss_logger.add_value(identifier=PREDICTED_GOAL,
                                            value=self._train_predict_goal(agent=self._agent))
            self.epoch += 1
            epoch_this_training += 1
        print("training finished!")
