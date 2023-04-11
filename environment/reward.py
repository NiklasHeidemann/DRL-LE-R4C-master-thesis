from abc import abstractmethod
from typing import Dict

from pettingzoo.utils.conversions import AgentID
from typing_extensions import Protocol
import numpy as np

from environment.generator import PositionIndex
from environment.stats import Stats


class ComputeReward(Protocol):

    @abstractmethod
    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats)->Dict[AgentID, float]:
        ...

class TwoCoopComputeReward(ComputeReward):
    _positive_reward: float = 1
    _negative_reward: float = -0.1

    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats)->Dict[AgentID, float]:
        colors_on_cell_by_agent = {agent_id: np.where(grid[position]!=0)[0] for agent_id, position in agent_positions.items()}
        assert len(colors_on_cell_by_agent)==2
        colors_1, colors_2 = list(colors_on_cell_by_agent.values())
        reward = self._negative_reward
        for color in colors_1:
            if color in colors_2:
                reward += self._positive_reward
                break
        return {agent_id: reward for agent_id in stats.agent_ids }

class SingleComputeReward(ComputeReward):
    _positive_reward: float = 1
    _negative_reward: float = -0.1

    def __call__(self, grid: np.ndarray, agent_positions: Dict[AgentID, PositionIndex], stats: Stats)->Dict[AgentID, float]:
        colors_on_cell_by_agent = {agent_id: np.where(grid[position]!=0)[0] for agent_id, position in agent_positions.items()}
        assert len(colors_on_cell_by_agent)==1
        reward = self._negative_reward if len(colors_on_cell_by_agent[stats.agent_ids[0]]) == 0 else self._positive_reward
        return {agent_id: reward for agent_id in stats.agent_ids }