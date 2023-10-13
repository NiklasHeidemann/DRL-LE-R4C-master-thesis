from typing import List, Optional

import numpy as np

from environment.env import CoopGridWorld


class EnvBatcher:

    def __init__(self, env: CoopGridWorld, batch_size: int):
        self._envs = [env.copy() for _ in range(batch_size)]
        self._stats = env._stats

    def reset(self, mask: List[bool], observation_array: np.ndarray)->np.ndarray:
        new_observation_array = observation_array.copy()
        for index, (env, m) in enumerate(zip(self._envs, mask)):
            if m:
                new_observation_array[index], _ = env.reset()
        return new_observation_array

    def step(self, actions_batch: np.ndarray):
        obs = np.zeros(shape=(len(self._envs), self._stats.recurrency, len(self._stats.agent_ids), self._stats.observation_dimension))
        reward = np.zeros(shape=(len(self._envs), len(self._stats.agent_ids)))
        terminated = np.zeros(shape=(len(self._envs)))
        truncated = np.zeros(shape=(len(self._envs)))
        for index, env in enumerate(self._envs):
            obs[index], reward[index], terminated[index], truncated[index] = env.step(actions=actions_batch[index])
        return obs, reward, terminated, truncated

    def reset_all(self):
        return np.array([env.reset()[0] for env in self._envs])

    def render(self, index: int):
        return self._envs[index].render()

    def get_envs(self, mask: Optional[np.ndarray]= None)->List[CoopGridWorld]:
        if mask is None:
            return self._envs
        return [env for take, env in zip(mask,self._envs) if take]

    @property
    def env_types(self)->np.ndarray:
        return np.array([env.current_type for env in self._envs])

    @property
    def size(self)->int:
        return len(self._envs)