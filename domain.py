from typing import Tuple, Dict

import numpy as np

ACTIONS = ["HOLD", "UP", "DOWN", "LEFT", "RIGHT"]

# If set to True, the environment will be rendered. This means that the test episodes will be displayed in a window
# during training. Furthermore, the displayed images will be saved in the logs directory
# This is useful for debugging and visualization, but slows down training significantly.

RENDER = False

"""
Returns all positions that are visible from the given position, assuming a Manhattan distance of 2.
"""
def visible_positions_13(x_y):
    return [
            (x_y[0]+1,x_y[1]+1),(x_y[0]+1,x_y[1]),(x_y[0]+1,x_y[1]-1),
            (x_y[0],x_y[1]+1),(x_y[0],x_y[1]),(x_y[0],x_y[1]-1),
            (x_y[0]-1,x_y[1]+1),(x_y[0]-1,x_y[1]),(x_y[0]-1,x_y[1]-1),
            (x_y[0]+2,x_y[1]),(x_y[0]-2,x_y[1]),(x_y[0],x_y[1]+2),(x_y[0],x_y[1]-2)
        ]

"""
Returns all positions that are visible from the given position, assuming a Manhattan distance of 1.
"""
def visible_positions_5(x_y):
    return [
            (x_y[0]+1,x_y[1]),
            (x_y[0],x_y[1]+1),(x_y[0],x_y[1]),(x_y[0],x_y[1]-1),
            (x_y[0]-1,x_y[1]),
        ]

# Type aliases
ENV_TYPE = int
AgentID = str
PositionIndex = Tuple[int,int]
RenderSave = Tuple[np.ndarray, Dict[AgentID, PositionIndex], Dict[AgentID, str], Dict[AgentID, np.ndarray], int, Dict[AgentID, int]]
RenderSaveExtended = Tuple[RenderSave, np.ndarray, Dict[AgentID, Tuple[float,float]]]
