from abc import abstractmethod
from collections import defaultdict
from threading import Thread
from typing import List, Dict

import numpy as np
import tensorflow as tf
from typing_extensions import Protocol

from domain import ACTIONS, RenderSaveExtended
from language.predictGoal import TrainPredictGoal
from utils.loss_logger import TEST_RETURNS, LossLogger, TEST_SOCIAL_RETURNS
from plotting.plots import plot_multiple
from training.Agent import Agent
from training.ExperienceReplayBuffer import ExperienceReplayBuffer


class Trainer(Protocol):

    def _init(self, environment, agent: Agent, plotting: bool, replay_buffer: ExperienceReplayBuffer, run_name:str, from_save: bool, metrics: Dict[int,List[str]]):
        self._loss_logger = LossLogger(run_name=run_name)
        self._environment = environment
        self._agent = agent
        self._last_render_as_list: List[List[RenderSaveExtended]] = []
        self._replay_buffer = replay_buffer
        self._run_name = run_name
        self._plotting = plotting
        self._train_predict_goal = TrainPredictGoal(environment=environment)
        if from_save:
            self.epoch = self._agent.load_models(name="", run_name=run_name)
            self._loss_logger.load(path="logger")
        else:
            self.epoch = 0
            for smooth_factor, metric_names in metrics.items():
                self._loss_logger.add_lists(metric_names, smoothed=smooth_factor)

    def _get_current_render_save(self, observation_array: np.ndarray, action_probs: np.ndarray)->RenderSaveExtended:
        values = self._agent.get_values(tf.expand_dims(observation_array, axis=0))
        values_by_agent = {agent: (values[0, index], 0) for index, agent in
                           enumerate(range(self._environment.stats.number_of_agents))}
        return ((self._environment.render(), action_probs, values_by_agent))

    @abstractmethod
    def train(self, num_epochs: int, render: bool)->LossLogger:
        ...

    def test(self, n_samples: int, render: bool)->(float, float):
        returns = []
        social_returns = []
        render_save: List[RenderSaveExtended] = []
        self._last_render_as_list.clear()
        for index in range(n_samples):
            observation_array, _ = self._environment.reset()
            action_probs_or_log_probs = defaultdict(lambda: np.zeros(shape=(len(ACTIONS))))
            return_ = 0
            social_reward = 0
            while True:
                if render:
                    render_save.append(self._get_current_render_save(observation_array=observation_array, action_probs=action_probs_or_log_probs))
                (actions_array, new_observation_array, reward_array, done), action_probs_or_log_probs = self._agent.act(
                    observation_array, deterministic=True, env=self._environment)
                return_ += sum(reward_array)
                social_reward += sum(self._agent.compute_social_reward(observation_prime=np.expand_dims(new_observation_array,axis=0),deterministic=True))
                if done:
                    returns.append(return_)
                    social_returns.append(social_reward)
                    if render:
                        render_save.append(self._get_current_render_save(observation_array=observation_array,
                                                                         action_probs=action_probs_or_log_probs))
                        self._last_render_as_list.append(render_save)
                        render_save = []
                    print(
                        f"TEST: Finished test episode {index} with a return of {return_} after {self._environment.stats.time_step} time steps.")
                    break
                observation_array = new_observation_array
        avg_return = np.mean(returns)
        avg_social_return = np.mean(social_returns)
        print(f"TEST: The average return is {avg_return}")
        return avg_return, avg_social_return

    def _every_few_epochs(self, render:bool):
        self._agent.save_models(name="", run_name=self._run_name, current_epoch=self.epoch)
        return_, social_return = self.test(n_samples=10, render=render)
        self._loss_logger.add_values({TEST_RETURNS: return_, TEST_SOCIAL_RETURNS   : social_return})
        self._loss_logger.save(path="logger")
        if self._plotting:
            thread = Thread(target=plot_multiple, args=[self._run_name, self._loss_logger.all_smoothed(), self.epoch])
            thread.start()