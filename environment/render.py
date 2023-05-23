import os
import time
from pathlib import Path
from typing import Mapping, Tuple, List, Dict

import pygame
import numpy as np

from environment.env import RenderSave, RenderSaveExtended, _map_communication_to_str
from environment.generator import PositionIndex
from params import ACTIONS

window = pygame.display.set_mode((1000, 500))
pygame.font.init()
def render(save: RenderSave, action_probs: np.ndarray, name: str, episode_index: int, max_q_value: Dict[str, Tuple[float,float]], log_dir: str):
    grid, agent_positions, last_agent_movements, last_communication, timestep = save
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
        text_field = my_font.render(f"{id}:     {ACTIONS[np.argmax(movement)]}", True, background_color)
        window.blit(text_field, (550,(index+1)*100+20))
        text_field = my_font.render(f"com:     {_map_communication_to_str(last_communication[id])}", True, background_color)
        window.blit(text_field, (550, (index+1)*100+40))
        text_field = my_font.render("max_q_val:   {:.2f}".format(max_q_value[id][0]), True, background_color)
        window.blit(text_field, (550, (index+1)*100+60))
        q_value_index = max_q_value[id][1]
        best_act = ACTIONS[q_value_index] if q_value_index<len(ACTIONS) else "com"
        text_field = my_font.render(f"best_act:   {best_act}", True, background_color)
        window.blit(text_field, (550, (index+1)*100+80))
        for action_index in range(len(ACTIONS)):
            text_field = my_font.render(f"{ACTIONS[action_index]}:     {'{:.2f}'.format(action_probs[index, action_index])}", True, background_color)
            window.blit(text_field, (750, (index+1)*100+20+action_index*20))
    pygame.display.update()
    pygame.image.save(window, log_dir+f"/{episode_index}.png")

def render_episode(render_saves: List[RenderSaveExtended], name:str)->None:
    log_dir = f"logs/{name}_{len(render_saves)}"
    os.mkdir(log_dir)
    for index, (save, action_probs, max_q_value) in enumerate(render_saves):
        render(save=save, action_probs=action_probs, name=name, episode_index=index, max_q_value=max_q_value, log_dir=log_dir)
        for _ in range(40):
            time.sleep(0.01)

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