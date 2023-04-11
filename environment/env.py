import random
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

import pygame as pygame
from babyrobot.envs.lib import GridLevel
import matplotlib.pyplot as plt

import numpy as np
from ipycanvas import Canvas
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ActionDict, ObsDict, AgentID

from params import TIME_STEPS
from environment.generator import WorldGenerator, PositionIndex
from environment.stats import Stats
from environment.reward import ComputeReward

RenderSave = Tuple[np.ndarray, Dict[AgentID, PositionIndex], Dict[AgentID, str], int]
RenderSaveExtended = Tuple[RenderSave, np.ndarray]

class CoopGridWorld(ParallelEnv):
    _grid: np.ndarray = None
    _stats: Stats = Stats()
    _agent_positions: Dict[AgentID, PositionIndex] = None
    _communications: List[Dict[AgentID, np.ndarray]] = None
    _last_agent_actions: List[Dict[AgentID, str]] = None
    _last_observations: Dict[AgentID, deque[np.ndarray]] = None
    def __init__(self, generator: WorldGenerator, compute_reward: ComputeReward):
        self._generator = generator
        self._compute_reward = compute_reward
    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:
        self._grid, self._agent_positions, self._stats = self._generator(last_stats=self._stats)
        self._communications = []
        self._last_agent_actions = [{agent_id:"-" for agent_id in self._stats.agent_ids}]
        self._communications.append({agent_id: np.zeros(shape=(self._stats.number_communication_channels*self._stats.size_vocabulary)) for agent_id in self._stats.agent_ids})
        self._last_observations = {agent_id: deque([self._obs_dict[agent_id]]*TIME_STEPS) for agent_id in self._stats.agent_ids }
        return self.obs


    def _obs_for_agent(self,agent_id: str)->np.ndarray:
        visible_positions = self.stats.visible_positions(self._agent_positions[agent_id])
        grid_observation = np.reshape(newshape=(-1,),a=np.array([self._grid[position] if self._is_valid_position(position) else (np.zeros(shape=(self._stats.values_per_field)) -1) for position in visible_positions]))
        communication_observation = np.concatenate([self._communications[-1][agent_id]]+[communication for com_agent_id, communication in self._communications[-1].items() if com_agent_id!=agent_id])

        return np.concatenate([grid_observation, communication_observation, self._stats.stats_observation])

    def _is_valid_position(self, position: Tuple[int, int])->bool:
        return position[0]>=0 and position[0]<len(self._grid) and position[1]>=0 and position[1]<len(self._grid[0])
    @property
    def _obs_dict(self)->Dict[AgentID, np.ndarray]:
        return {agent_id: self._obs_for_agent(agent_id) for agent_id in self._stats.agent_ids}

    @property
    def obs(self):
        return {agent_id: np.reshape(obs, newshape=(TIME_STEPS, -1)) for agent_id, obs in  self._last_observations.items()}
    def seed(self, seed=None):
        random.seed(seed)

    def step(self, actions: Dict[AgentID, Tuple[str, Optional[List[int]]]]) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        self._stats.time_step+=1
        self._communications.append({})
        self._last_agent_actions.append({})
        for agent_id, (movement, communication) in actions.items():
            self._move_agent(agent_id=agent_id, movement=movement)
            self._last_agent_actions[-1][agent_id]=movement
            self._communications[-1][agent_id] = np.array(communication)
        reward_dict = self._compute_reward(grid=self._grid, agent_positions=self._agent_positions, stats=self._stats)
        is_terminated = max(reward_dict.values())>0
        is_truncated = self._stats.time_step >= self._stats.max_time_step
        for agent_id, obs in self._obs_dict.items():
            self._last_observations[agent_id].append(obs)
            self._last_observations[agent_id].popleft()
        return (
            self.obs,
            reward_dict,
            {agent_id: is_terminated for agent_id in self._stats.agent_ids},
            {agent_id: is_truncated for agent_id in self._stats.agent_ids},
            {agent_id: {} for agent_id in self._stats.agent_ids}, #info_dict
        )
    def _move_agent(self, agent_id: str, movement: str):
        old_x, old_y = self._agent_positions[agent_id]
        if movement == "HOLD":
            x, y = old_x, old_y
        elif movement == "UP":
            x, y = old_x - 1, old_y
        elif movement == "DOWN":
            x, y = old_x + 1, old_y
        elif movement == "LEFT":
            x, y = old_x, old_y - 1
        elif movement == "RIGHT":
            x, y = old_x, old_y + 1
        elif movement == "JUMP":
            x, y = random.randint(0, self.stats.grid_size -1), random.randint(0, self.stats.grid_size -1)
        else:
            raise Exception(f"Invalid command {movement} for agent {agent_id} in timestep {self._stats.time_step}")
        if not self._is_valid_position(position=(x, y)):
            x, y = old_x, old_y
        self._agent_positions[agent_id] = (x, y)

    def render(self) -> RenderSave:
        grid = '\n'+'\n'.join([''.join([self._map_cell_to_char(pos=(x,y)) for y in range(self._grid.shape[1])]) for x in range(self._grid.shape[0])])
        communication = '\t'.join([f"{agent_id}: '{self._map_communication_to_str(communication=com)}'" for agent_id, com in self._communications[-1].items()])
        output = grid + '\n' + communication
        #print(output)
        render_save = self._grid.copy(), self._agent_positions.copy(), self._last_agent_actions[-1], self.stats.time_step
        return render_save
    def state(self) -> np.ndarray:
        pass

    @property
    def stats(self)-> Stats:
        return self._stats

    def _map_communication_to_str(self, communication: np.ndarray )->str:
        chars = []
        for index in range(0, len(communication), self.stats.size_vocabulary):
            token = communication[index:index+self.stats.size_vocabulary]
            if sum(token) == 0:
                chars.append('-')
            else:
                assert sum(token) == 1
                chars.append(chr(ord("a")+communication.argmax()))
        return ''.join(chars)
    def _map_cell_to_char(self, pos: Tuple[int, int])->str:
        agent_on_spot = pos in self._agent_positions.values()
        cell = self._grid[pos]
        if sum(cell) == 0:
            return "*" if agent_on_spot else "_"
        assert sum(cell)==1
        character = chr(ord("a") + np.argmax(cell))
        return character.upper() if agent_on_spot else character
