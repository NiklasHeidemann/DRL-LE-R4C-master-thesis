from abc import abstractmethod

from typing_extensions import Protocol
import numpy as np

from grid_world import GOAL_POSITION


class GridWorldAgent(Protocol):
    @abstractmethod
    def pick_action(self, state:np.ndarray)->str:
        ...

class ManhattanWalker(GridWorldAgent):
    def pick_action(self, state: np.ndarray)->str:
        x_pos, y_pos = np.where(state == 1)
        if len(x_pos) == 0:
            return "HOLD"
        assert len(x_pos) == 1 and len(y_pos) == 1
        own_pos = x_pos[0], y_pos[0]
        goal_pos = GOAL_POSITION
        if own_pos[0]>goal_pos[0]:
            return "UP"
        if own_pos[0]<goal_pos[0]:
            return "DOWN"
        if own_pos[1]<goal_pos[1]:
            return "RIGHT"
        if own_pos[1]>goal_pos[1]:
            return "LEFT"
        else:
            raise Exception("Bug")

