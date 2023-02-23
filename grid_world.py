import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import gymnasium
import tensorflow as tf
import numpy as np
from pettingzoo.utils.env import ParallelEnv, ActionDict, AgentID
from tensorflow.python.util.tf_decorator import rewrap

ObsDict = Dict[str, np.ndarray]

POS_REWARD = 10
NEG_REWARD = -0.1
GOAL_POSITION = (4,4)
GRID_SHAPE = (6,6)
MAX_TIMESTEP = 20

ACTIONS = ["HOLD", "LEFT", "RIGHT", "UP", "DOWN"]
#map_agent_to_starting_positions = {"0_0":(0,0)}
def map_agent_to_starting_positions(agent_id: str)->Tuple[int,int]:
    return (int(agent_id.split("_")[0]),int(agent_id.split("_")[1]))

class DemoGridWorld(ParallelEnv):
    """
    N agents. 5x5 grid. Agent starts at fixed positions, goal position is (4,4). Reward is +10 for being on the goal position.
    Actions are left, right, up, down, hold. If two agents try to move to the same position, a randomly selected one starts at his starting position (as soon as that becomes free). Same if they move out of bounds or reach the goal position.
    """
    _grid: np.ndarray = None
    _seed: float = None
    _time_step: int = 0
    possible_agents = [f"{a}_{b}" for a,b in [(a,b) for a in range(0,6) for b in range(0,6)] if a!=4 or b!=4]

    def __init__(self, possible_agents: Optional[List[str]] = None):
        if possible_agents is not None:
            self.possible_agents = possible_agents

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:
        self._grid = np.zeros(shape=GRID_SHAPE)
        self._time_step = 0
        self.agents = self.possible_agents
        for agent in self.agents:
            assert self._grid[map_agent_to_starting_positions(agent)] == 0, "No two agents can start on the same positions"
            self._grid[map_agent_to_starting_positions(agent)] = self.possible_agents.index(agent) + 1
        return {agent: self._obs_for_agent(agent) for agent in self.agents}

    def seed(self, seed=None):
        self._seed = seed
        tf.random.set_seed(seed)
        random.seed(seed)

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        self._time_step+=1
        target_positions: Dict[Tuple[int,int], List[AgentID]] = defaultdict(list)
        spawners = []
        for agent,action in actions.items():
            x_pos, y_pos = np.where(self._grid == self.possible_agents.index(agent) + 1)
            if len(x_pos) == 0:
                spawners.append(agent)
                continue
            assert len(x_pos)==1 and len(y_pos)==1
            agent_pos = x_pos[0], y_pos[0]
            if action == "HOLD":
                new_pos = agent_pos
            elif action == "UP":
                new_pos = (agent_pos[0] -1, agent_pos[1])
            elif action == "DOWN":
                new_pos = (agent_pos[0] +1, agent_pos[1])
            elif action == "LEFT":
                new_pos = (agent_pos[0] , agent_pos[1]-1)
            elif action == "RIGHT":
                new_pos = (agent_pos[0], agent_pos[1]+1)
            else:
                raise Exception(f"Illegal move {action} for agent {agent}")
            if new_pos[0]<0 or new_pos[0]>=len(self._grid) or new_pos[1]<0 or new_pos[1]>=len(self._grid[0]):
                spawners.append(agent)
            else:
                target_positions[new_pos].append(agent)
        new_grid = np.zeros_like(self._grid)
        for position, agent_list in target_positions.items():
            chosen_index = random.randint(0, len(agent_list)-1)
            new_grid[position] = self.possible_agents.index(agent_list.pop(chosen_index))+1
            spawners.extend(agent_list)
        for agent in spawners:
            agent_pos = map_agent_to_starting_positions(agent)
            if new_grid[agent_pos]==0:
                new_grid[agent_pos]=self.possible_agents.index(agent)+1
                #print(f"respawn {agent}")
        reward_dict: Dict[str, float] = {agent: POS_REWARD if new_grid[GOAL_POSITION]==self.possible_agents.index(agent)+1 else NEG_REWARD for agent in self.agents}
        if new_grid[GOAL_POSITION]!=0:
            print(f"Agent {new_grid[GOAL_POSITION]} reached the goal")
            new_grid[GOAL_POSITION] = 0 # agent will respawn next iteration
        self._grid = new_grid
        obs: Dict[str, np.ndarray] = {agent: self._obs_for_agent(agent) for agent in self.agents}
        termination_dict: Dict[str, bool] = {agent: self._time_step>=MAX_TIMESTEP for agent in self.agents}
        truncation_dict: Dict[str, bool] = {agent: False for agent in self.agents}
        info_dict: Dict[str, Dict] = {agent: {} for agent in self.agents}
        return obs, reward_dict, termination_dict, truncation_dict, info_dict

    def render(self) -> None | np.ndarray | str | List:
        return self._grid

    def state(self) -> Tuple[int, np.ndarray]:
        return (self._time_step, self._grid)

    def _obs_for_agent(self, agent_id: str)->np.ndarray:
        obs = np.zeros_like(self._grid)
        np.place(obs, self._grid != 0, -1)
        np.place(obs, self._grid == self.possible_agents.index(agent_id) + 1, 1)
        return np.concatenate([obs.flatten(),[self._time_step]])

    @property
    def observation_space_shape(self)->Tuple[int,...]:
        return GRID_SHAPE
    @property
    def observation_space_dim(self)->int:
        return GRID_SHAPE[0]*GRID_SHAPE[1]+1

    @property
    def action_space_shape(self)->Tuple[int,...]:
        return (5,)