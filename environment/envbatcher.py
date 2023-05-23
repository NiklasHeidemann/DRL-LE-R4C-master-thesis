from pettingzoo import ParallelEnv

from environment.env import CoopGridWorld


class EnvBatcher:

    def __init__(self, env: CoopGridWorld, batch_size: int):
        self._envs = [env.copy() for _ in range(batch_size)]

    def step(self, ):
        ...

