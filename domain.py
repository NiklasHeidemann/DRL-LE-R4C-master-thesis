from typing import Tuple, Dict

import numpy as np

ACTIONS = ["HOLD", "UP", "DOWN", "LEFT", "RIGHT"]
ENV_TYPE = int
AgentID = str

RENDER = False

def visible_positions_13(x_y):
    return [
            (x_y[0]+1,x_y[1]+1),(x_y[0]+1,x_y[1]),(x_y[0]+1,x_y[1]-1),
            (x_y[0],x_y[1]+1),(x_y[0],x_y[1]),(x_y[0],x_y[1]-1),
            (x_y[0]-1,x_y[1]+1),(x_y[0]-1,x_y[1]),(x_y[0]-1,x_y[1]-1),
            (x_y[0]+2,x_y[1]),(x_y[0]-2,x_y[1]),(x_y[0],x_y[1]+2),(x_y[0],x_y[1]-2)
        ]

visible_positions_5 =         lambda x_y: [
            (x_y[0]+1,x_y[1]),
            (x_y[0],x_y[1]+1),(x_y[0],x_y[1]),(x_y[0],x_y[1]-1),
            (x_y[0]-1,x_y[1]),
        ]

PositionIndex = Tuple[int,int]
RenderSave = Tuple[np.ndarray, Dict[AgentID, PositionIndex], Dict[AgentID, str], Dict[AgentID, np.ndarray], int, Dict[AgentID, int]]
RenderSaveExtended = Tuple[RenderSave, np.ndarray, Dict[AgentID, Tuple[float,float]]]
