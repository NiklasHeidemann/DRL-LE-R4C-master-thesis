import math
# list files in directory
import os
from pathlib import Path

import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
from utils.loss_logger import ACTOR_LOSS, CRITIC_LOSS, COM_ENTROPY, RETURNS, V_VALUES, STD_ADVANTAGE, TEST_RETURNS, \
    AVG_ADVANTAGE, KLD, PREDICTED_GOAL, SOCIAL_REWARD, TEST_SOCIAL_RETURNS

ROOT_PATH = Path("/home/nemo/pycharmProjects/adversarial/saves/exp_1")
SAVE_PATH = Path("/home/nemo/pycharmProjects/adversarial/saves/all")
KEYS = [ACTOR_LOSS, CRITIC_LOSS, COM_ENTROPY, RETURNS, V_VALUES, STD_ADVANTAGE, TEST_RETURNS, AVG_ADVANTAGE, KLD, PREDICTED_GOAL, SOCIAL_REWARD, TEST_SOCIAL_RETURNS]

def group_image():
    for key in KEYS:

        images = [(Image.open(ROOT_PATH / path), path.split(f"{key}.png")[0]) for path in os.listdir(ROOT_PATH) if
                  path.endswith(f"{key}.png")]

        grid_size = math.ceil(len(images) ** 0.5)
        fig = make_subplots(rows=grid_size, cols=grid_size, subplot_titles=[name for _, name in images],
                            horizontal_spacing=0.01, vertical_spacing=0.01)

        for index, (image, _) in enumerate(images):
            fig.add_trace(go.Image(z=image), row=index // grid_size + 1, col=index % grid_size + 1)

        fig.update_layout(height=600 * grid_size, width=800 * grid_size, title_text=f"{key}")
        # remove the x and y ticks
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        # remove the legend and axes
        fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False)
        fig.update_layout(font=dict(size=30))
        print("grouping plots for", key)
        fig.write_image(str(SAVE_PATH / f"{key}.png"))


if __name__ == '__main__':
    group_image()