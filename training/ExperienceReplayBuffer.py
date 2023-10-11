from abc import abstractmethod
from typing import Any, Dict, Union, Optional, Tuple

import numpy as np
import tensorflow as tf
from typing_extensions import Protocol

ACTION_KEY = "action"
STATE_KEY = "state"
REWARD_KEY = "reward"

class ExperienceReplayBuffer(Protocol):
    """
    Experience Replay Buffer
    state_dims, action_dims, max_size=1000000, batch_size=256
    :param state_dims: Dimensions of the state space
    :param action_dims: Dimensions of the action space
    :param max_size=1000000: Size of the replay buffer
    :param batch_size=256: Minibatch size for each gradient update
    """

    def _init(self, state_dims, action_dims, agent_number: float, max_size:int, batch_size:int, time_steps: Optional[int]):
        self._max_size = max_size
        self._batch_size = batch_size
        self._size = 0
        self._current_position = 0
        self._memories = {}
        self._memories[ACTION_KEY] = np.zeros((self._max_size, agent_number,action_dims))
        self._memories[REWARD_KEY] = np.zeros((self._max_size, agent_number))
        self._action_memory = np.zeros((self._max_size, agent_number,action_dims))
        self._reward_memory = np.zeros((self._max_size, agent_number))
        if time_steps is not None:
            self._memories[STATE_KEY] = np.zeros((self._max_size, time_steps, agent_number,*state_dims))
        else:
            self._memories[STATE_KEY] = np.zeros((self._max_size, agent_number,*state_dims))

    def size(self):
        return self._size

    def ready(self):
        return self._size >= self._batch_size


    def add_transition_batch(self, inputs: Dict[str, Union[np.ndarray,tf.Tensor]])->None:
        sample_size = list(inputs.values())[0].shape[0]
        if self._current_position + sample_size > self._max_size:
            self.add_transition_batch(inputs={key: value_array[:self._max_size-self._current_position] for key, value_array in inputs.items()})
            self.add_transition_batch(inputs={key: value_array[self._max_size-self._current_position:] for key, value_array in inputs.items()})
            return
        index = (self._current_position, self._current_position + sample_size)
        for key, memory in self._memories.items():
            memory[index[0]:index[1]] = inputs[key]
        self._size = min(self._max_size, self._size + sample_size)
        self._current_position = (self._current_position + sample_size) % self._max_size

    def sample_batch(self, batch_indices = None):
        if batch_indices is None:
            batch_indices = np.random.choice(self._size, self._batch_size, replace=False)
        sampled_memories = {key: tf.convert_to_tensor(memory[batch_indices], dtype=tf.float32) for key, memory in self._memories.items()}
        return sampled_memories

    def clear(self)->None:
        self._size = 0
        self._current_position = 0

    def get_all_repeated(self, repetitions: int):
        batch_indices = np.array(range(self._size
                                       ))
        np.random.shuffle(batch_indices)
        return tf.data.Dataset.from_tensor_slices(self.sample_batch(batch_indices=batch_indices)).repeat(repetitions).batch(batch_size=self._batch_size)


