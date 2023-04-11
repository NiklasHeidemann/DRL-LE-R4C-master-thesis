from typing import Callable, Tuple, List

import numpy as np

from params import ACTIONS


class Stats:
    grid_size: int = None
    number_of_agents: int = None
    number_of_objects: int = None
    number_of_used_colors: int = None
    time_step: int = None
    max_time_step: int = None
    values_per_field: int = None
    number_communication_channels: int = None
    size_vocabulary: int = None
    visible_positions: Callable[[Tuple[int,int]],List[Tuple[int,int]]]

    @property
    def agent_ids(self) -> List[str]:
        return [str(index) for index in range(self.number_of_agents)]

    @property
    def stats_observation(self)->np.ndarray:
        #return np.array([self.grid_size, self.number_of_agents, self.number_of_objects, self.number_of_used_colors, self.time_step])
        return np.array([])

    @property
    def number_of_visible_positions(self)->int:
        return len(self.visible_positions((0,0)))
    @property
    def observation_dimension(self)->int:
        return self.number_of_visible_positions*self.values_per_field + self.size_vocabulary*self.number_communication_channels*self.number_of_agents + len(self.stats_observation)

    @property
    def action_dimension(self)->int:
        return len(ACTIONS) + self.number_communication_channels * (1+self.size_vocabulary)
    def new_stats_from_old_stats(self)->"Stats":
        new_stats = Stats()
        new_stats.max_time_step = self.max_time_step
        new_stats.values_per_field = self.values_per_field
        new_stats.number_communication_channels = self.number_communication_channels
        new_stats.size_vocabulary = self.size_vocabulary
        new_stats.visible_positions = self.visible_positions
        return new_stats



