import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from environment.env import CoopGridWorld
from environment.generator import RandomWorldGenerator
from environment.reward import RaceReward
from params import GRID_SIZE_RANGE, NUMBER_OF_OBJECTS_TO_PLACE_RANGE, OBJECT_COLOR_RANGE, MAX_TIME_STEP, \
    AGENT_DROPOUT_PROBS, VISIBLE_POSITIONS, TIME_STEPS

env = CoopGridWorld(generator=RandomWorldGenerator(seed=99, grid_size_range=GRID_SIZE_RANGE,
                                                      number_of_agents=3,number_of_objects_to_place=NUMBER_OF_OBJECTS_TO_PLACE_RANGE,
                                                      number_of_object_colors_range=OBJECT_COLOR_RANGE, agent_dropout_probs=0.,
                                                      max_time_step=MAX_TIME_STEP),compute_reward=RaceReward(pos_reward=2, neg_reward=-0.1, communism=True),xenia_permanence=False,xenia_lock=True)
env.stats.number_communication_channels = 0
env.stats.size_vocabulary = 4
env.stats.visible_positions = VISIBLE_POSITIONS
env.stats.recurrency = TIME_STEPS
env.reset()
model = RecurrentPPO("MlpLstmPolicy", env=env, verbose=1)
model.learn(5000)

vec_env = model.get_env()
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
print(mean_reward)

model.save("ppo_recurrent")
del model # remove to demonstrate saving and loading

model = RecurrentPPO.load("ppo_recurrent")

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones
    vec_env.render("human")