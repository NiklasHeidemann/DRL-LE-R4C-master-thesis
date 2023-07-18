import multiprocessing

import tensorflow as tf
import numpy as np
from typing_extensions import override



class PPOExperienceReplayBuffer:
    """
    Experience Replay Buffer
    state_dims, action_dims, max_size=1000000, batch_size=256
    :param state_dims: Dimensions of the state space
    :param action_dims: Dimensions of the action space
    :param max_size=1000000: Size of the replay buffer
    :param batch_size=256: Minibatch size for each gradient update
    """

    def __init__(self, state_dims, action_dims, agent_number: float, max_size=1000000, batch_size=256):
        self._max_size = max_size
        self._batch_size = batch_size
        self._size = 0
        self._current_position = 0
        self._state_memory = np.zeros((self._max_size, agent_number,*state_dims))
        self._action_memory = np.zeros((self._max_size, agent_number,action_dims))
        self._reward_memory = np.zeros((self._max_size, agent_number))
        self._done_memory = np.zeros((self._max_size, 1), dtype=bool)
        self._advantage_memory = np.zeros((self._max_size, agent_number))
        self._prob_old_policy_memory = np.zeros((self._max_size, agent_number))

    def size(self):
        return self._size

    def ready(self):
        return self._size >= self._batch_size

    def _add_state_transitions(self, state):
        self._state_memory[self._current_position] = state

    def _add_state_transition_batch(self, state):
        self._state_memory[self._current_position: self._current_position+state.shape[0]] = state

    def add_transition(self, state, action, reward, state_, done, advantage, prob_old_policy):
            self._action_memory[self._current_position] = action
            self._reward_memory[self._current_position] = reward
            self._done_memory[self._current_position] = done
            self._add_state_transitions(state=state,state_=state_)
            self._advantage_memory[self._current_position] = advantage
            self._prob_old_policy_memory[self._current_position] = prob_old_policy
            if self._size < self._max_size:
                self._size += 1
            self._current_position = (self._current_position + 1) % self._max_size

    def add_transition_batch(self, state, action, advantage, prob_old_policy, reward):
            if self._current_position + state.shape[0] > self._max_size:
                self.add_transition_batch(state=state[:self._max_size-self._current_position],
                                          action=action[:self._max_size-self._current_position],
                                          reward=reward[:self._max_size-self._current_position],
                                          #state_=state_[:self._max_size-self._current_position],
                                          #done=done[:self._max_size-self._current_position],
                                          advantage=advantage[:self._max_size-self._current_position],
                                          prob_old_policy=prob_old_policy[:self._max_size-self._current_position]
                                          )
                self.add_transition_batch(state=state[self._max_size-self._current_position:],
                        action=action[self._max_size-self._current_position:],advantage=advantage[self._max_size-self._current_position:],prob_old_policy=prob_old_policy[self._max_size-self._current_position:], reward=reward[self._max_size-self._current_position])
                return
            index = (self._current_position,self._current_position+state.shape[0])
            self._action_memory[index[0]:index[1]] = action
            self._reward_memory[index[0]:index[1]] = reward
            #self._done_memory[index[0]:index[1]] = tf.expand_dims(done,axis=1)
            self._add_state_transition_batch(state=state)
            self._advantage_memory[index[0]:index[1]] = advantage
            self._prob_old_policy_memory[index[0]:index[1]] = prob_old_policy
            self._size = min(self._max_size, self._size + state.shape[0])
            self._current_position = (self._current_position + state.shape[0]) % self._max_size


    def sample_batch(self, batch_indices = None):
        if batch_indices is None:
            batch_indices = np.random.choice(self._size, self._batch_size, replace=False)
        states = tf.convert_to_tensor(self._state_memory[batch_indices], dtype=tf.float32)
        actions = tf.convert_to_tensor(self._action_memory[batch_indices], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self._reward_memory[batch_indices], dtype=tf.float32)
        advantages = tf.convert_to_tensor(self._advantage_memory[batch_indices], dtype=tf.float32)
        prob_olds = tf.convert_to_tensor(self._prob_old_policy_memory[batch_indices], dtype=tf.float32)
        return states, actions, rewards, advantages, prob_olds

    def get_all_repeated(self, repetitions: int):
        batch_indices = np.array(range(self._size
                                       ))
        np.random.shuffle(batch_indices)
        return tf.data.Dataset.from_tensor_slices(self.sample_batch(batch_indices=batch_indices)).repeat(repetitions).batch(batch_size=self._batch_size)
    def clear(self)->None:
        self._size = 0
        self._current_position = 0

class PPORecurrentExperienceReplayBuffer(PPOExperienceReplayBuffer):
    def __init__(self, state_dims, time_steps: int, action_dims, agent_number: float, max_size=1000000, batch_size=256):
        super().__init__(state_dims=state_dims, action_dims=action_dims, agent_number=agent_number, max_size=max_size, batch_size=batch_size)
        self._state_memory = np.zeros((self._max_size, time_steps,agent_number,*state_dims))

    @override
    def _add_state_transitions(self, state):
            self._state_memory[self._current_position] = state
