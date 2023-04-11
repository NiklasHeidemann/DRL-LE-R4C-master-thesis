import os
import time
from pathlib import Path
from time import sleep
from typing import Mapping, Tuple, List

import pygame
import numpy as np

from environment.env import RenderSave, RenderSaveExtended
from environment.generator import PositionIndex
from params import ACTIONS

window = pygame.display.set_mode((1000, 500))
pygame.font.init()
def render(save: RenderSave, action_probs: np.ndarray, name: str, episode_index: int):
    grid, agent_positions, last_agent_movements, timestep = save
    global window
    background_color = (0, 0, 0)
    textfield_color = (220,220,220)
    cell_colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    default_cell_color = (220,220,220)
    window.fill(background_color)  # fill background with color
    # make draw calls

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            color = default_cell_color if sum(grid[row,col])==0 else cell_colors[grid[row,col].argmax()]
            pygame.draw.rect(surface=window, color=color, rect=(50*col, 50*row, 49, 49))

    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    for id, (row,col) in agent_positions.items():
        text_field =my_font.render(id,True,background_color)
        window.blit(text_field, (col*50+10, row*50+10))

    pygame.draw.rect(surface=window,color=textfield_color, rect=(500, 0, 500,500))
    text_field = my_font.render(f"timestep:     {timestep}", True, background_color)
    window.blit(text_field, (550,20))
    for index, (id, movement) in enumerate(last_agent_movements.items()):
        text_field = my_font.render(f"{id}:     {movement}", True, background_color)
        window.blit(text_field, (550,(index+1)*100+20))
        for action_index in range(len(ACTIONS)):
            text_field = my_font.render(f"{ACTIONS[action_index]}:     {'{:.2f}'.format(action_probs[id][0][action_index])}", True, background_color)
            window.blit(text_field, (750, (index+1)*100+20+action_index*20))
    pygame.display.update()
    pygame.image.save(window, f"logs/{name}/{episode_index}.png")

def render_episode(render_saves: List[RenderSaveExtended], name:str)->None:
    log_dir = f"logs/{name}"
    os.mkdir(log_dir)
    for index, (save, action_probs) in enumerate(render_saves):
        render(save=save, action_probs=action_probs, name=name, episode_index=index)
        sleep(0.4)

def render_permanently(render_saves_as_list: List[List[RenderSaveExtended]])->None:
    clean_logs()
    counter = 0
    while True:
        time.sleep(2)
        if len(render_saves_as_list)>0:
            episode = render_saves_as_list[-1]
            render_episode(episode, str(counter))
            counter+=1

def clean_logs():
    for dir in os.listdir("logs"):
        assert (Path("logs") / dir).is_dir(), dir
        for file in os.listdir(f"logs/{dir}"):
            os.remove(f"logs/{dir}/{file}")
        os.rmdir(f"logs/{dir}")