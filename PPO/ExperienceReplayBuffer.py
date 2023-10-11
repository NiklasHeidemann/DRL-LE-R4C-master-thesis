import multiprocessing
from typing import Dict, Optional, Union, Tuple

import tensorflow as tf
import numpy as np
from typing_extensions import override

from training.ExperienceReplayBuffer import ExperienceReplayBuffer, ACTION_KEY, REWARD_KEY, STATE_KEY

ADVANTAGE_KEY = "advantage"
PROB_OLD_KEY = "prob_old"

class PPOExperienceReplayBuffer(ExperienceReplayBuffer):
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
        self._init(state_dims=state_dims,action_dims=action_dims,agent_number=agent_number,max_size=max_size,batch_size=batch_size,time_steps=time_steps)
        self._memories[ADVANTAGE_KEY] = np.zeros((max_size, agent_number))
        self._memories[PROB_OLD_KEY] = np.zeros((max_size, agent_number))


