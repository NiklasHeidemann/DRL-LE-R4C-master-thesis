import random
from abc import abstractmethod
from typing import Tuple, Dict, List

from pettingzoo.utils.env import AgentID
from typing_extensions import Protocol
import numpy as np

from environment.stats import Stats
from params import NUMBER_OF_AGENTS, NUMBER_OF_OBJECTS_TO_PLACE_RANGE, MAX_OBJECT_COLOR_RANGE, \
    GRID_SIZE_RANGE, MAX_TIME_STEP

Grid = np.ndarray
PositionIndex = Tuple[int,int]
AgentPositions = Dict[AgentID, PositionIndex]
Type = str


class WorldGenerator(Protocol):


    @abstractmethod
    def __call__(self, last_stats: Stats)->Tuple[Grid, AgentPositions, Stats, Type]:
        ...

    @property
    @abstractmethod
    def types(self)->List[str]:
        ...

class MultiGenerator(WorldGenerator):

    def __init__(self, generators: List[WorldGenerator]):
        self._generators = generators
    def __call__(self, last_stats: Stats)->Tuple[Grid, AgentPositions, Stats, Type]:
        return self._generators[random.randrange(len(self._generators))](last_stats=last_stats)

    @property
    def types(self)->List[str]:
        return [elem for list_ in [generator.types for generator in self._generators] for elem in list_]

class RandomWorldGenerator(WorldGenerator):
    _grid_size_range: Tuple[int, int] = GRID_SIZE_RANGE # inclusive interval
    _number_of_object_colors_range: Tuple[int, int] = (1,MAX_OBJECT_COLOR_RANGE) # inclusive interval
    _number_of_objects_to_place: Tuple[int,int] = NUMBER_OF_OBJECTS_TO_PLACE_RANGE
    _number_of_agents: int = NUMBER_OF_AGENTS

    def __init__(self, seed: int):
        random.seed(seed)
    def __call__(self, last_stats: Stats)->Tuple[Grid, AgentPositions, Stats, Type]:
        stats = last_stats.new_stats_from_old_stats()
        stats.values_per_field = self._number_of_object_colors_range[1]
        stats.time_step = 0
        stats.max_time_step = MAX_TIME_STEP
        stats.grid_size = random.randint(self._grid_size_range[0], self._grid_size_range[1])
        grid = np.zeros(shape=(stats.grid_size, stats.grid_size, stats.values_per_field))
        grid_with_objects = self._place_objects(grid=grid, stats=stats)
        agent_positions = self._spawn_agents(stats=stats)
        return grid_with_objects, agent_positions, stats, self.types[0]
    def _spawn_agents(self, stats: Stats)->AgentPositions:
        stats.number_of_agents = self._number_of_agents
        agent_positions = {}
        for agent_id in stats.agent_ids:
            agent_positions[agent_id] = (random.randint(0, stats.grid_size-1), random.randint(0, stats.grid_size-1))
        return agent_positions
    def _place_objects(self, grid: np.ndarray, stats: Stats)->np.ndarray:
        stats.number_of_objects = random.randint(int(self._number_of_objects_to_place[0]*stats.grid_size*stats.grid_size), int(self._number_of_objects_to_place[1]*stats.grid_size*stats.grid_size))
        stats.number_of_used_colors = random.randint(self._number_of_object_colors_range[0], self._number_of_object_colors_range[1])
        for _ in range(stats.number_of_objects):
            object_color = random.randint(0, stats.number_of_used_colors-1)
            position = self._pick_free_position(grid=grid)
            grid[(position[0],position[1], object_color)] = 1
        return grid
    def _pick_free_position(self, grid: np.ndarray)->PositionIndex:
        position = None
        while position is None or max(grid[(position[0],position[1])])!=0:
            position = (random.randint(0, len(grid)-1), random.randint(0,len(grid[0])-1))
        return position

    @property
    def types(self)->List[str]:
        return ["rwg"]

class ChoiceWorldGenerator(WorldGenerator):
    _grid_size: int = 10
    _number_of_object_colors: int= MAX_OBJECT_COLOR_RANGE
    _number_of_objects_to_place_per_agent: int = 4
    _number_of_agents: int = 2

    def __init__(self, seed: int):
        random.seed(seed)
    def __call__(self, last_stats: Stats)->Tuple[Grid, AgentPositions, Stats, Type]:
        stats = last_stats.new_stats_from_old_stats()
        stats.values_per_field = self._number_of_object_colors
        stats.time_step = 0
        stats.max_time_step = 3
        stats.grid_size = self._grid_size
        grid = np.zeros(shape=(stats.grid_size, stats.grid_size, stats.values_per_field))
        grid_with_objects = self._place_objects(grid=grid, stats=stats)
        agent_positions = self._spawn_agents(stats=stats)
        return grid_with_objects, agent_positions, stats, self.types[0]
    def _spawn_agents(self, stats: Stats)->AgentPositions:
        stats.number_of_agents = self._number_of_agents
        agent_positions = {'0':(2,2),'1':(7,7)}
        return agent_positions
    def _place_objects(self, grid: np.ndarray, stats: Stats)->np.ndarray:
        stats.number_of_objects = self._number_of_objects_to_place_per_agent*2
        stats.number_of_used_colors = self._number_of_object_colors
        indexes_1 = [(1,1),(1,3),(3,1),(3,3)]
        indexes_2 = [(5,7),(7,9),(7,5),(9,7)]
        for x,y in indexes_1:
            object_color = random.randint(0, stats.number_of_used_colors-1)
            grid[(x,y,object_color)] = 1
        mode = random.randrange(0,stats.number_of_used_colors+0)
        for x,y in indexes_2:
            if mode<stats.number_of_used_colors:
                grid[(x,y,mode)] = 1
            else:
                object_color = random.randint(0, stats.number_of_used_colors-1)
                grid[(x,y,object_color)] = 1
        return grid

    @property
    def types(self)->List[str]:
        return ["chwg"]