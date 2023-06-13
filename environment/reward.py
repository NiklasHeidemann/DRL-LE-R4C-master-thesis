from abc import abstractmethod
from typing import Dict, Optional

from pettingzoo.utils.conversions import AgentID
from typing_extensions import Protocol
import numpy as np

from environment.generator import PositionIndex
from environment.stats import Stats


class ComputeReward(Protocol):

    @abstractmethod
    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats, agents_locked: Optional[Dict[AgentID, int]])->np.ndarray:
        ...

class RaceReward(ComputeReward):
    _positive_reward: float
    _negative_reward: float
    _communism: bool

    def __init__(self, pos_reward: float, neg_reward: float, communism: bool):
        self._negative_reward = neg_reward
        self._positive_reward = pos_reward
        self._communism = communism

    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats, agents_locked: Optional[Dict[AgentID, int]])->np.array:
        if agents_locked is None:
            colors_on_cell_by_agent = {agent_id: np.where(grid[position]!=0)[0] for agent_id, position in agent_positions.items()}
        else:
            colors_on_cell_by_agent = {agent_id: [] if value == -1 else [value] for agent_id, value in agents_locked.items()}
        assert len(colors_on_cell_by_agent)==stats.number_of_agents
        color_list = list([color[0] if len(color)==1 else None for color in colors_on_cell_by_agent.values()])
        rewards = np.zeros(stats.number_of_agents) + self._negative_reward
        for index, color in enumerate(color_list):
            if color is not None and (color in color_list[index+1:] or color in color_list[:index]):
                rewards[index] = self._positive_reward
        if self._communism:
            return np.zeros(stats.number_of_agents) + max(rewards)
        return rewards

class TwoCoopComputeReward(ComputeReward):
    _positive_reward: float
    _negative_reward: float

    def __init__(self, pos_reward: float, neg_reward: float):
        self._negative_reward = neg_reward
        self._positive_reward = pos_reward

    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats, agents_locked: Optional[Dict[AgentID, int]])->np.array:
        colors_on_cell_by_agent = {agent_id: np.where(grid[position]!=0)[0] for agent_id, position in agent_positions.items()}
        if agents_locked is not None:
            colors_on_cell_by_agent = {agent_id: [] if value == -1 else [value] for agent_id, value in agents_locked.items()}
        assert len(colors_on_cell_by_agent)==2
        colors_1, colors_2 = list(colors_on_cell_by_agent.values())
        reward = self._negative_reward
        for color in colors_1:
            if color in colors_2:
                reward += self._positive_reward
                break
        return np.array([reward for _ in stats.agent_ids])

class AtLeastOneComputeReward(ComputeReward):
    _positive_reward: float
    _negative_reward: float

    def __init__(self, pos_reward: float, neg_reward: float):
        self._negative_reward = neg_reward
        self._positive_reward = pos_reward

    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats, agents_locked: Optional[Dict[AgentID, int]])->np.ndarray:
        colors_on_cell_by_agent = {agent_id: np.where(grid[position]!=0)[0] for agent_id, position in agent_positions.items()}
        if agents_locked is not None:
            colors_on_cell_by_agent = {agent_id: [] if value == -1 else [value] for agent_id, value in agents_locked.items()}
        assert len(colors_on_cell_by_agent)==2
        colors_1, colors_2 = list(colors_on_cell_by_agent.values())
        reward = self._positive_reward if len(colors_1)+len(colors_2)>0 else self._negative_reward
        return np.array([reward for _ in stats.agent_ids])

class FirstOneComputeReward(ComputeReward):
    _positive_reward: float
    _negative_reward: float

    def __init__(self, pos_reward: float, neg_reward: float):
        self._negative_reward = neg_reward
        self._positive_reward = pos_reward

    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats, agents_locked: Optional[Dict[AgentID, int]])->np.ndarray:
        colors_on_cell_by_agent = {agent_id: np.where(grid[position]!=0)[0] for agent_id, position in agent_positions.items()}
        if agents_locked is not None:
            colors_on_cell_by_agent = {agent_id: [] if value == -1 else [value] for agent_id, value in agents_locked.items()}
        assert len(colors_on_cell_by_agent)==2
        colors_1, colors_2 = list(colors_on_cell_by_agent.values())
        reward = self._positive_reward if len(colors_1)>0 else self._negative_reward
        return np.array([reward for _ in stats.agent_ids])


class SingleComputeReward(ComputeReward):
    _positive_reward: float
    _negative_reward: float

    def __init__(self, pos_reward: float, neg_reward: float):
        self._negative_reward = neg_reward
        self._positive_reward = pos_reward

    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats, agents_locked: Optional[Dict[AgentID, int]])->np.ndarray:
        colors_on_cell_by_agent = {agent_id: np.where(grid[position]!=0)[0] for agent_id, position in agent_positions.items()}
        if agents_locked is not None:
            colors_on_cell_by_agent = {agent_id: [] if value == -1 else [value] for agent_id, value in agents_locked.items()}
        assert len(colors_on_cell_by_agent)==1
        reward = self._negative_reward if len(colors_on_cell_by_agent[stats.agent_ids[0]]) == 0 else self._positive_reward
        return np.array([reward for _ in stats.agent_ids])