import random
from functools import partial

from SAC.GenericMLPs1D import create_policy_network, create_q_network
from SAC.SoftActorCriticAgent import SACAgent
from environment.env import CoopGridWorld
from environment.generator import RandomWorldGenerator
from environment.reward import TwoCoopComputeReward, SingleComputeReward, AtLeastOneComputeReward, FirstOneComputeReward
import tensorflow as tf
import matplotlib

from params import MAX_TIME_STEP, NUMBER_COMMUNICATION_CHANNELS, SIZE_VOCABULARY, VISIBLE_POSITIONS, \
    NUMBER_OF_BIG_LAYERS, NUMBER_OF_AGENTS, MAX_REPLAY_BUFFER_SIZE, SEED, EPOCHS, PRE_SAMPLING_STEPS, \
    ENVIRONMENT_STEPS_PER_TRAINING, SELF_PLAY, RECURRENT, FROM_SAVE
from plots import delete_old_plots

matplotlib.use("agg")
#delete_old_plots()
random.seed(SEED)
tf.random.set_seed(SEED)

compute_reward = SingleComputeReward() if NUMBER_OF_AGENTS == 1 else FirstOneComputeReward()

env = CoopGridWorld(generator=RandomWorldGenerator(seed=SEED), compute_reward=compute_reward)
env.stats.max_time_step = MAX_TIME_STEP
env.stats.number_communication_channels = NUMBER_COMMUNICATION_CHANNELS
env.stats.size_vocabulary = SIZE_VOCABULARY
env.stats.visible_positions = VISIBLE_POSITIONS
env.reset()

sac_agent = SACAgent(environment=env, from_save=FROM_SAVE, self_play=SELF_PLAY, agent_ids=env.stats.agent_ids, state_dim=(env.stats.observation_dimension,), action_dim=env.stats.action_dimension, recurrent=RECURRENT,
                     max_replay_buffer_size=MAX_REPLAY_BUFFER_SIZE,
                     actor_network_generator=partial(create_policy_network, state_dim=env.stats.observation_dimension,
                                                     action_dim=env.stats.action_dimension, number_of_big_layers=NUMBER_OF_BIG_LAYERS),
                     critic_network_generator=partial(create_q_network, state_dim=env.stats.observation_dimension,
                                                      action_dim=env.stats.action_dimension, number_of_big_layers=NUMBER_OF_BIG_LAYERS))
sac_agent._critic_1.summary()
sac_agent._actor.summary() if SELF_PLAY else sac_agent._actors["0"].summary()
sac_agent.test(n_samples=2, verbose_samples=0)
sac_agent.train(epochs=EPOCHS, pre_sampling_steps=PRE_SAMPLING_STEPS, environment_steps_before_training=ENVIRONMENT_STEPS_PER_TRAINING) # question, intuition about epoch number, and replaybuffer parameters
sac_agent.test(n_samples=20, verbose_samples=0)