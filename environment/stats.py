from typing import Callable, Tuple, List

import numpy as np

from domain import ACTIONS

"""
Class handling all the parameters of an environment instance. Some will be constant over all episodes, others will change with each call of the generator.
"""
class Stats:
    grid_size: int = None
    number_of_agents: int = None
    placed_agents: int = None
    number_of_objects: int = None
    number_of_used_colors: int = None
    time_step: int = None
    max_time_step: int = None
    values_per_field: int = None
    number_communication_channels: int = None
    size_vocabulary: int = None
    recurrency: int = None
    visible_positions: Callable[[Tuple[int,int]],List[Tuple[int,int]]]

    @property
    def agent_ids(self) -> List[str]:
        return [str(index) for index in range(self.number_of_agents)]

    @property
    def stats_observation(self)->np.ndarray:
        return np.array([self.max_time_step/30, self.time_step/self.max_time_step*0, self.number_of_used_colors/self.size_vocabulary])

    @property
    def number_of_visible_positions(self)->int:
        return len(self.visible_positions((0,0)))
    @property
    def observation_dimension(self)->int:
        return (self.number_of_visible_positions*self.values_per_field
            +(self.size_vocabulary+1)*self.number_communication_channels*self.number_of_agents
            + len(self.stats_observation)
            + 2 +#coordinates
            + self.values_per_field # locked color
                )

    """
    returns the index in the observation of agent_index that corresponds to the beginning of the communication of 
    agent speaker_index 
    """
    def index_of_communication_in_observation(self, agent_index: int, speaker_index: int) -> int:
        base_position = self.number_of_visible_positions*self.values_per_field
        if agent_index == speaker_index:
            return base_position
        actual_index = speaker_index if agent_index<speaker_index else speaker_index+1
        return base_position + actual_index * (self.size_vocabulary+1)*self.number_communication_channels

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
        new_stats.recurrency = self.recurrency
        return new_stats



