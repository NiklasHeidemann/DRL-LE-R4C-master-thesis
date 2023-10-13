from abc import abstractmethod
from typing import Dict, Optional

import numpy as np
from typing_extensions import Protocol

from domain import AgentID, PositionIndex
from environment.stats import Stats


class ComputeReward(Protocol):
    """
    Returns the reward for each agent for a given state.

    :grid: The attribute from CoopGridWorld
    :agent_positions: The attribute from CoopGridWorld
    :stats: The attribute from CoopGridWorld
    :agents_locked: if None, only look at the current positions. If true, instead use the locked colors from this Dict

    :returns the reward for each agent
    """
    @abstractmethod
    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats, agents_locked: Optional[Dict[AgentID, int]])->np.ndarray:
        ...

"""
If two or more agents are on the same color (or have this color locked if agents_locked is not None), then all agents
on this color receive a positive reward. All agents that are not, a negative. If communism is True, then all agents 
receive the maximum (not the sum) of the rewards of the single agents.
"""
class RaceReward(ComputeReward):
    _positive_reward: float
    _negative_reward: float
    _communism: bool

    def __init__(self, pos_reward: float, neg_reward: float, communism: bool):
        self._negative_reward = neg_reward
        self._positive_reward = pos_reward
        self._communism = communism

    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats, agents_locked: Optional[Dict[AgentID, int]])->np.array:
        rewards = np.zeros(stats.number_of_agents) + self._negative_reward
        if agents_locked is None:
            colors_on_cell_by_agent = {agent_id: np.where(grid[position]!=0)[0] for agent_id, position in agent_positions.items()}
        else:
            colors_on_cell_by_agent = {agent_id: [] if value == -1 else [value] for agent_id, value in agents_locked.items()}
        color_list = list([color[0] if len(color)==1 else None for color in colors_on_cell_by_agent.values()])
        for index, color in enumerate(color_list):
            if color is not None and (color in color_list[index+1:] or color in color_list[:index]):
                rewards[index] = self._positive_reward
        if self._communism:
            return np.zeros(stats.number_of_agents) + max(rewards)
        return rewards

"""
Supports only exactly two agents. Both agents receive a reward if any of them is on a colored field.
"""
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

"""
Supports only exactly two agents. All agents receive a positive reward exactly if the first agent is on a field with
a color.
"""
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

"""
Supports only exactly one agent. The agent receives a positive reward, if it is on a cell with a color.
"""
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