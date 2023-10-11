import multiprocessing
from typing import Optional, Dict, Union, Tuple

import tensorflow as tf
import numpy as np
from typing_extensions import override

from training.ExperienceReplayBuffer import ExperienceReplayBuffer, ACTION_KEY, REWARD_KEY, STATE_KEY

STATE_PRIME_KEY = "state_prime"
DONE_KEY = "done"

class SACExperienceReplayBuffer(ExperienceReplayBuffer):
    """
    Experience Replay Buffer
    state_dims, action_dims, max_size=1000000, batch_size=256
    :param state_dims: Dimensions of the state space
    :param action_dims: Dimensions of the action space
    :param max_size=1000000: Size of the replay buffer
    :param batch_size=256: Minibatch size for each gradient update
    """

    def __init__(self, state_dims, action_dims, agent_number: float, max_size:int, batch_size:int, time_steps: Optional[int]):
        super().__init__()
        self._init(state_dims=state_dims, action_dims=action_dims, agent_number=agent_number, max_size=max_size,
                         batch_size=batch_size, time_steps=time_steps)
        if time_steps is None:
            self._memories[STATE_PRIME_KEY] = np.zeros((max_size, agent_number,*state_dims))
        else:
            self._memories[STATE_PRIME_KEY] = np.zeros((max_size, time_steps, agent_number, *state_dims))
        self._memories[DONE_KEY] = np.zeros((self._max_size, 1), dtype=bool)

