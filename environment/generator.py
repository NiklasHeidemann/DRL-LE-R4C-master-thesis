import random
from abc import abstractmethod
from typing import Tuple, Dict, List, Optional

import numpy as np
from typing_extensions import Protocol

from domain import AgentID, PositionIndex
from environment.stats import Stats

Grid = np.ndarray
AgentPositions = Dict[AgentID, Optional[PositionIndex]]
Type = str


class WorldGenerator(Protocol):
    """
    Create a new world instance (variables that can be used by CoopGridWorld).

    last_stats: Stats of the last episode. Values that are not specified by the WorldGenerator will be copied from this.
        If this is the first episode: Specify the necessary values in a novel stats object.
    """

    @abstractmethod
    def __call__(self, last_stats: Stats) -> Tuple[Grid, AgentPositions, Stats, Type]:
        ...

    """
    types of environments that can be generated.
    type is a property of World instances (that is CoopGridWorld) 
    """
    @property
    @abstractmethod
    def types(self) -> List[str]:
        ...


"""
Combines multiple generators to one. When called, will simply call one of its known generators at random. 
"""


class MultiGenerator(WorldGenerator):

    def __init__(self, generators: List[WorldGenerator]):
        self._generators = generators

    """
    see documentation of WorldGenerator above
    """
    def __call__(self, last_stats: Stats) -> Tuple[Grid, AgentPositions, Stats, Type]:
        return self._generators[random.randrange(len(self._generators))](last_stats=last_stats)

    @property
    def types(self) -> List[str]:
        return [elem for list_ in [generator.types for generator in self._generators] for elem in list_]


"""
Most versatile grid world generator that was used for the most relevant experiment. Simply creates a world with colored
cells ("objects") where all agents can run around. 
"""


class RandomWorldGenerator(WorldGenerator):
    _grid_size_range: Tuple[int, int]
    _number_of_object_colors_range: Tuple[int, int]
    _number_of_objects_to_place: Tuple[float, float]
    _number_of_agents: int
    _agent_dropout_probs: float
    _max_time_step: int

    """
    grid_size_range: Min and max of the edge size (both width and height) of the grid. A random value in the inclusive
        interval of [min, max] will be sampled for each world instance.
    number_of_object_colors_range: Min and max of the number of colors that exist in a single world instance. Max equals
        the overall number of colors. A random value in the inclusive interval of [min, max] will be sampled for each
        world instance. All of the colors will be equally frequent in the world instances.
    number_of_objects_to_place: Min and max of the percent of fields that will have one color. A random value in the
        inclusive interval of [min, max] will be sampled for each world instance. If e.g. 0.4 then 40% of the cells
        will have one color.
    number_of_agents: Size of the set of agents that can be spawned in the world 
    agent_dropout_probs: Probability that one random agent will not be spawned for any given world instance
    max_time_step: Number of time steps after which the episode should be truncated. 
    """

    def __init__(self,
                 grid_size_range: Tuple[int, int],
                 number_of_object_colors_range: Tuple[int, int],
                 number_of_objects_to_place: Tuple[float, float],
                 number_of_agents: int,
                 agent_dropout_probs: float,
                 max_time_step: int,
                 seed: int
                 ) -> None:
        super().__init__()
        self._max_time_step = max_time_step
        self._number_of_agents = number_of_agents
        self._agent_dropout_probs = agent_dropout_probs
        self._number_of_object_colors_range = number_of_object_colors_range
        self._grid_size_range = grid_size_range
        self._number_of_objects_to_place = number_of_objects_to_place
        random.seed(seed)

    """
    see documentation of WorldGenerator above
    """
    def __call__(self, last_stats: Stats) -> Tuple[Grid, AgentPositions, Stats, Type]:
        stats = last_stats.new_stats_from_old_stats()
        stats.values_per_field = self._number_of_object_colors_range[1]
        stats.time_step = 0
        stats.max_time_step = self._max_time_step
        stats.grid_size = random.randint(self._grid_size_range[0], self._grid_size_range[1])
        grid = np.zeros(shape=(stats.grid_size, stats.grid_size, stats.values_per_field))
        grid_with_objects = self._place_objects(grid=grid, stats=stats)
        agent_positions = self._spawn_agents(stats=stats)
        return grid_with_objects, agent_positions, stats, self.types[0]

    def _spawn_agents(self, stats: Stats) -> AgentPositions:
        stats.number_of_agents = self._number_of_agents
        agent_positions = {}
        for agent_id in stats.agent_ids:
            agent_positions[agent_id] = (random.randint(0, stats.grid_size - 1), random.randint(0, stats.grid_size - 1))
        if random.random() > self._agent_dropout_probs:
            stats.placed_agents = stats.number_of_agents
        else:
            stats.placed_agents = stats.number_of_agents - 1
            agent_positions[stats.agent_ids[random.randrange(len(agent_positions))]] = None
        return agent_positions

    def _place_objects(self, grid: np.ndarray, stats: Stats) -> np.ndarray:
        stats.number_of_objects = random.randint(
            int(self._number_of_objects_to_place[0] * stats.grid_size * stats.grid_size),
            int(self._number_of_objects_to_place[1] * stats.grid_size * stats.grid_size)
        )
        stats.number_of_used_colors = random.randint(self._number_of_object_colors_range[0],
                                                     self._number_of_object_colors_range[1])
        used_color_set = random.sample(range(self._number_of_object_colors_range[1]), k=stats.number_of_used_colors)
        for _ in range(stats.number_of_objects):
            object_color = used_color_set[random.randint(0, stats.number_of_used_colors - 1)]
            position = self._pick_free_position(grid=grid)
            grid[(position[0], position[1], object_color)] = 1
        return grid

    def _pick_free_position(self, grid: np.ndarray) -> PositionIndex:
        position = None
        while position is None or max(grid[(position[0], position[1])]) != 0:
            position = (random.randint(0, len(grid) - 1), random.randint(0, len(grid[0]) - 1))
        return position

    @property
    def types(self) -> List[str]:
        return ["rwg"]

"""
Creates environment of fixed size with two agents. Both agents are placed far from each other, so they can not reach
each others' colored cells ("objects"). One of the agents has only objects of the same color available. The other one
has one object of that color and multiple objects of random colors available. The max_time_step is chosen such that the 
agents only have time to try a single object before the episode is truncated.
"""
class ChoiceWorldGenerator(WorldGenerator):
    _grid_size: int = 10
    _number_of_objects_to_place_per_agent: int = 4
    _max_time_step: int = 4
    _number_of_agents: int
    _number_of_object_colors: int

    """
    number_of_object_colors:  Min and max of the number of colors that exist in a single world instance. Max equals
        the overall number of colors. A random value in the inclusive interval of [min, max] will be sampled for each
        world instance. All of the colors will be equally frequent in the world instances.
    """
    def __init__(self, seed: int, object_color_range: Tuple[int, int], number_of_agents: int) -> None:
        random.seed(seed)
        self._number_of_object_colors = object_color_range[1]
        self._number_of_agents = number_of_agents

    """
    see documentation of WorldGenerator above
    """
    def __call__(self, last_stats: Stats) -> Tuple[Grid, AgentPositions, Stats, Type]:
        stats = last_stats.new_stats_from_old_stats()
        stats.values_per_field = self._number_of_object_colors
        stats.time_step = 0
        stats.max_time_step = self._max_time_step
        stats.grid_size = self._grid_size
        grid = np.zeros(shape=(stats.grid_size, stats.grid_size, stats.values_per_field))
        grid_with_objects = self._place_objects(grid=grid, stats=stats)
        agent_positions = self._spawn_agents(stats=stats)
        return grid_with_objects, agent_positions, stats, self.types[0]

    def _spawn_agents(self, stats: Stats) -> AgentPositions:
        stats.number_of_agents = self._number_of_agents
        agent_positions = {'0': (2, 2), '1': (3, 8)}
        if self._number_of_agents == 3:
            agent_positions['2'] = (8, 4)
        return agent_positions

    def _place_objects(self, grid: np.ndarray, stats: Stats) -> np.ndarray:
        stats.number_of_objects = self._number_of_objects_to_place_per_agent * 2
        stats.number_of_used_colors = self._number_of_object_colors
        indexes_1 = [(1, 1), (1, 3), (3, 1), (3, 3)]
        indexes_2 = [(1,8)]
        indexes_3 = [(8,2)]
        used_colors = set()
        for x, y in indexes_1:
            object_color = random.randrange(0, stats.number_of_used_colors)
            while stats.number_of_used_colors>=len(indexes_1) and object_color in used_colors:
                object_color = random.randrange(0, stats.number_of_used_colors)
            grid[(x, y, object_color)] = 1
            used_colors.add(object_color)
        assert len(used_colors) == 4
        if len(used_colors) == 1:
            mode_agent_2 = list(used_colors)[0]
            mode_agent_3 = list(used_colors)[0]
        else:
            mode_agent_2, mode_agent_3 = None, None
            agent_2_gets_guaranteed_fit = self._number_of_agents==2 or random.random() < 2/3
            agent_3_gets_guaranteed_fit = not agent_2_gets_guaranteed_fit or random.random() < 0.5
            while mode_agent_2 not in used_colors:
                mode_agent_2 = random.randrange(0, stats.number_of_used_colors)
                if not agent_2_gets_guaranteed_fit:
                    break
            if self._number_of_agents == 3:
                while mode_agent_3 not in used_colors or mode_agent_3==mode_agent_2:
                    mode_agent_3 = random.randrange(0, stats.number_of_used_colors)
                    if not agent_3_gets_guaranteed_fit:
                        break
        assert mode_agent_2 in used_colors or mode_agent_3 in used_colors
        for x, y in indexes_2:
            grid[(x, y, mode_agent_2)] = 1
        if self._number_of_agents == 3:
            for x, y in indexes_3:
                grid[(x, y, mode_agent_3)] = 1
        return grid

    @property
    def types(self) -> List[str]:
        return ["chwg"]

class ManhattanGenerator(WorldGenerator):
    _grid_size: int = 6
    _number_of_objects_to_place_per_agent: int = 4
    _max_time_step: int = 6
    _number_of_agents: int
    _number_of_object_colors: int

    """
    number_of_object_colors:  Min and max of the number of colors that exist in a single world instance. Max equals
        the overall number of colors. A random value in the inclusive interval of [min, max] will be sampled for each
        world instance. All of the colors will be equally frequent in the world instances.
    """
    def __init__(self, seed: int, object_color_range: Tuple[int, int], number_of_agents: int) -> None:
        random.seed(seed)
        self._number_of_object_colors = object_color_range[1]
        self._number_of_agents = number_of_agents

    """
    see documentation of WorldGenerator above
    """
    def __call__(self, last_stats: Stats) -> Tuple[Grid, AgentPositions, Stats, Type]:
        stats = last_stats.new_stats_from_old_stats()
        stats.values_per_field = self._number_of_object_colors
        stats.time_step = 0
        stats.max_time_step = self._max_time_step
        stats.grid_size = self._grid_size
        grid = np.zeros(shape=(stats.grid_size, stats.grid_size, stats.values_per_field))
        grid_with_objects = self._place_objects(grid=grid, stats=stats)
        agent_positions = self._spawn_agents(stats=stats)
        return grid_with_objects, agent_positions, stats, self.types[0]

    def _spawn_agents(self, stats: Stats) -> AgentPositions:
        stats.number_of_agents = self._number_of_agents
        agent_positions = {'0': (3, 3)}
        return agent_positions

    def _place_objects(self, grid: np.ndarray, stats: Stats) -> np.ndarray:
        stats.number_of_objects = self._number_of_objects_to_place_per_agent * 2
        stats.number_of_used_colors = self._number_of_object_colors
        indexes = [(2,3), (3,2), (3,4), (4,3),(2,2), (2,4), (4,2), (4,4), (1,3), (3,1), (3,5), (5,3)]
        for x, y in indexes:
            grid[(x, y, 0)] = 1
        return grid

    @property
    def types(self) -> List[str]:
        return ["chwg"]
