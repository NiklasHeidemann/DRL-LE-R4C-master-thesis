import random
from collections import deque
from typing import List, Tuple, Dict, Optional

import gymnasium
import numpy as np
import tensorflow as tf

from domain import ACTIONS, ENV_TYPE, PositionIndex
from domain import AgentID, RenderSave
from environment.generator import WorldGenerator
from environment.reward import ComputeReward
from environment.stats import Stats

DEFAULT_COMMUNCIATIONS = lambda size_vocabulary, number_communication_channels: np.concatenate([[1.]+[0.]*size_vocabulary]*number_communication_channels) if number_communication_channels > 0 else []



class CoopGridWorld(gymnasium.Env):
    _grid: np.ndarray = None
    _stats: Stats = Stats()
    _agent_positions: Dict[AgentID, PositionIndex] = None
    _communications: List[Dict[AgentID, np.ndarray]] = None
    _last_agent_actions: List[Dict[AgentID, str]] = None
    _last_observations: deque[np.ndarray] = None
    _type: str = None
    _xenia_lock: bool = None
    _xenia_permanence: bool = None
    _agents_locked: Dict[AgentID, int] = None
    def __init__(self, generator: WorldGenerator, compute_reward: ComputeReward, xenia_lock: bool, xenia_permanence: bool):
        self._generator = generator
        self._compute_reward = compute_reward
        self._xenia_lock = xenia_lock
        self._xenia_permanence = xenia_permanence

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> np.ndarray:
        self._grid, self._agent_positions, self._stats, self._type = self._generator(last_stats=self._stats)
        self._communications = []
        self._last_agent_actions = [{agent_id:"-" for agent_id in self._stats.agent_ids}]
        self._communications.append({agent_id: DEFAULT_COMMUNCIATIONS(self._stats.size_vocabulary, self._stats.number_communication_channels) for agent_id in self._stats.agent_ids})
        self._agents_locked = {agent_id: -1 for agent_id in self._stats.agent_ids}
        #self._last_observations = {agent_id: deque([self._obs_dict[agent_id]]*TIME_STEPS) for agent_id in self._stats.agent_ids }
        self._last_observations = deque([self._obs_array]*self._stats.recurrency)
        return self.obs, {}


    def _obs_for_agent(self,agent_id: str)->np.ndarray:
        if self._agent_positions[agent_id] is None:
            return np.zeros(shape=(self._stats.observation_dimension))
        visible_positions = self.stats.visible_positions(self._agent_positions[agent_id])
        grid_observation = np.reshape(newshape=(-1,),a=np.array([self._grid[position] if self._is_valid_position(position) else (np.zeros(shape=(self._stats.values_per_field)) -1) for position in visible_positions]))
        communication_observation = np.concatenate([self._communications[-1][agent_id]]+[communication for com_agent_id, communication in self._communications[-1].items() if com_agent_id!=agent_id])
        coordinates = np.array([self._agent_positions[agent_id][0]/len(self._grid), self._agent_positions[agent_id][1]/len(self._grid[0])])*0 #todo
        selected_colour = tf.one_hot(self._agents_locked[agent_id],depth=self.stats.values_per_field) if self._xenia_lock else np.zeros(shape=(self.stats.values_per_field))
        return np.concatenate([grid_observation, communication_observation, self._stats.stats_observation, coordinates, selected_colour])

    def _is_valid_position(self, position: Tuple[int, int])->bool:
        return position[0]>=0 and position[0]<len(self._grid) and position[1]>=0 and position[1]<len(self._grid[0])
    @property
    def _obs_array(self)->np.ndarray:
        return np.array([self._obs_for_agent(agent_id) for agent_id in self._stats.agent_ids])

    @property
    def obs(self)->np.ndarray: # dimensions: TimeStep, AgentId, Observation
        return np.array(self._last_observations)
    def seed(self, seed=None):
        random.seed(seed)

    def step(self,
             actions: np.ndarray # dimensions: AgentID, Action+Communication
             ) -> Tuple[
        np.array, np.ndarray, bool, bool
    ]:
        self._stats.time_step+=1
        self._communications.append({})
        self._last_agent_actions.append({})
        for index, agent_id in enumerate(self._stats.agent_ids):
            if self._agent_positions[agent_id] is None:
                self._last_agent_actions[-1][agent_id] = actions[index,:len(ACTIONS)]
                self._communications[-1][agent_id] = DEFAULT_COMMUNCIATIONS(self._stats.size_vocabulary, self._stats.number_communication_channels)
            else:
                movement = actions[index,:len(ACTIONS)]
                self._move_agent(agent_id=agent_id, movement=movement)
                self._last_agent_actions[-1][agent_id]=movement
                self._communications[-1][agent_id] = np.array(actions[index,len(ACTIONS):])
        reward_array = self._compute_reward(grid=self._grid, agent_positions=self._agent_positions, stats=self._stats, agents_locked = self._agents_locked if self._xenia_lock else None)
        is_terminated = max(reward_array)>0
        is_truncated = (not is_terminated) and self._stats.time_step >= self._stats.max_time_step
        self._last_observations.append(self._obs_array)
        self._last_observations.popleft()
        return (
            self.obs,
            reward_array,
            is_terminated,
            is_truncated,
        )
    def _move_agent(self, agent_id: str, movement: np.ndarray):
        assert sum(movement)==1, movement
        old_x, old_y = self._agent_positions[agent_id]
        if movement[0] == 1:# "HOLD":
            x, y = old_x, old_y
        elif movement[1] == 1:#"UP":
            x, y = old_x - 1, old_y
        elif movement[2] == 1:#"DOWN":
            x, y = old_x + 1, old_y
        elif movement[3] == 1:#"LEFT":
            x, y = old_x, old_y - 1
        elif movement[4] == 1:#"RIGHT":
            x, y = old_x, old_y + 1
        else:
            raise Exception(f"Invalid command {movement} for agent {agent_id} in timestep {self._stats.time_step}")
        if not self._is_valid_position(position=(x, y)):
            x, y = old_x, old_y
        self._agent_positions[agent_id] = (x, y)
        if self._xenia_lock and ((not self._xenia_permanence) or self._agents_locked[agent_id] <0) and sum(self._grid[x,y])>0:
            self._agents_locked[agent_id] = np.argmax(self._grid[x,y])


    def render(self) -> RenderSave:
        grid = '\n'+'\n'.join([''.join([self._map_cell_to_char(pos=(x,y)) for y in range(self._grid.shape[1])]) for x in range(self._grid.shape[0])])
        communication = '\t'.join([f"{agent_id}: '{_map_communication_to_str(communication=com) if len(com)>0 else ''}'" for agent_id, com in self._communications[-1].items()])
        render_save = self._grid.copy(), self._agent_positions.copy(), self._last_agent_actions[-1], self._communications[-1], self.stats.time_step, self._agents_locked.copy()
        return render_save
    def state(self) -> np.ndarray:
        pass

    @property
    def stats(self)-> Stats:
        return self._stats

    def _get_agent_colors(self)->Dict[AgentID, int]:
        if self._xenia_lock:
            return self._agents_locked
    def _map_cell_to_char(self, pos: Tuple[int, int])->str:
        agent_on_spot = pos in self._agent_positions.values()
        cell = self._grid[pos]
        if sum(cell) == 0:
            return "*" if agent_on_spot else "_"
        assert sum(cell)==1
        character = chr(ord("a") + np.argmax(cell))
        return character.upper() if agent_on_spot else character

    def copy(self)->"CoopGridWorld":
        env =  CoopGridWorld(generator=self._generator, compute_reward=self._compute_reward, xenia_lock=self._xenia_lock, xenia_permanence=self._xenia_permanence)
        env._stats = self._stats
        return env

    @property
    def observation_space(self)->gymnasium.spaces.box.Box:
        return gymnasium.spaces.box.Box(low=-1, high=1, shape=[self.stats.recurrency, self._stats.number_of_agents,self.stats.observation_dimension], dtype=np.float32)

    @property
    def action_space(self)->gymnasium.spaces.MultiDiscrete:
        return gymnasium.spaces.MultiDiscrete([len(ACTIONS)] + [self._stats.size_vocabulary] * self._stats.number_communication_channels)

    @property
    def current_type(self)->ENV_TYPE:
        return self._stats.placed_agents

def _map_communication_to_str(communication: np.ndarray) -> str:
    chars = []
    size_vocabulary = int(len(communication)/sum(communication))
    assert int(size_vocabulary) == size_vocabulary
    for index in range(0, len(communication), int(size_vocabulary)):
        token = communication[index:index + size_vocabulary]
        index = communication.argmax()
        assert sum(token) == 1, f"{token}, {communication}"
        if index == 0:
            chars.append('-')
        else:
            chars.append(chr(ord("a") + index - 1))
    return ''.join(chars)
