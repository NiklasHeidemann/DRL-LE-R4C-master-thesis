from typing import List
import numpy as np
from pettingzoo import ParallelEnv

from environment.env import CoopGridWorld
from environment.stats import Stats
from params import TIME_STEPS


class EnvBatcher:

    def __init__(self, env: CoopGridWorld, batch_size: int):
        self._envs = [env.copy() for _ in range(batch_size)]
        self._stats = env._stats

    def reset(self, mask: List[bool], observation_array: np.ndarray)->np.ndarray:
        new_observation_array = observation_array.copy()
        for index, (env, m) in enumerate(zip(self._envs, mask)):
            if m:
                new_observation_array[index] = env.reset()
        return new_observation_array

    def step(self, actions_batch: np.ndarray):
        obs = np.zeros(shape=(len(self._envs), TIME_STEPS, len(self._stats.agent_ids), self._stats.observation_dimension))
        reward = np.zeros(shape=(len(self._envs), len(self._stats.agent_ids)))
        terminated = np.zeros(shape=(len(self._envs)))
        truncated = np.zeros(shape=(len(self._envs)))
        for index, env in enumerate(self._envs):
            obs[index], reward[index], terminated[index], truncated[index] = env.step(actions=actions_batch[index])
        return obs, reward, terminated, truncated

    def reset_all(self):
        return np.array([env.reset() for env in self._envs])

    def render(self, index: int):
        return self._envs[index].render()


