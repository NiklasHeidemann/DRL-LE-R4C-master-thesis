from functools import partial
from typing import Tuple, Dict, Any, Callable

from SAC.GenericMLPs1D import create_policy_network, create_q_network
from SAC.Trainer import Trainer
from environment.env import CoopGridWorld
from environment.generator import RandomWorldGenerator, ChoiceWorldGenerator, MultiGenerator
from environment.reward import ComputeReward, RaceReward
import random
import tensorflow as tf

class RunConfig:
    name: str = None
    VISIBLE_POSITIONS: Callable = None
    # network
    BATCH_SIZE = 128
    LAYER_SIZE = 32
    LEARNING_RATE = 0.001
    LSTM_SIZE = 32
    NUMBER_OF_BIG_LAYERS = 2
    RECURRENT = True
    SELF_PLAY = True
    TIME_STEPS = 10
    ALPHA = 0.06
    GAMMA = 0.99
    TARGET_ENTROPY = 1.

    # environment
    WORLD_GENERATOR = "random"  # multi or random or choice
    GRID_SIZE_RANGE = (4, 8)
    MAX_TIME_STEP = 30
    NUMBER_OF_AGENTS = 3
    COMMUNISM = True
    AGENT_DROPOUT_PROBS = 0.5 if NUMBER_OF_AGENTS == 3 else 0  # meaning with 0.5 probabilty the third agent is not placed
    NUMBER_OF_OBJECTS_TO_PLACE_RANGE = (0.08, 0.15)
    OBJECT_COLOR_RANGE = (1, 5)
    POS_REWARD = 2
    NEG_REWARD = -0.1
    REWARD_TYPE = "race"
    XENIA_LOCK = True
    XENIA_PERMANENCE = False

    # input
    NUMBER_COMMUNICATION_CHANNELS = 1
    SIZE_VOCABULARY = OBJECT_COLOR_RANGE[1]

    # buffer
    MAX_REPLAY_BUFFER_SIZE = 10000
    PRE_SAMPLING_STEPS = 10000

    # training
    ENVIRONMENT_STEPS_PER_TRAINING = 500
    TRAININGS_PER_TRAINING = 8
    EPOCHS = 200000
    SEED = 13
    ENV_PARALLEL = 32
    FROM_SAVE = False
    RENDER = False

    def __init__(self, params: Dict[str, Any]):
        for key, value in params.items():
            assert key in self.__dir__()
            setattr(self, key, value)
    def __call__(self):

        random.seed(self.SEED)
        tf.random.set_seed(self.SEED)
        if self.REWARD_TYPE == "race":
            self.compute_reward = RaceReward(pos_reward=self.POS_REWARD, neg_reward=self.NEG_REWARD, communism=self.COMMUNISM)
        else:
            raise NotImplemented()

        random_world_generator = RandomWorldGenerator(seed=self.SEED, grid_size_range=self.GRID_SIZE_RANGE,
                                                      number_of_agents=self.NUMBER_OF_AGENTS,number_of_objects_to_place=self.NUMBER_OF_OBJECTS_TO_PLACE_RANGE,
                                                      number_of_object_colors_range=self.OBJECT_COLOR_RANGE, agent_dropout_probs=self.AGENT_DROPOUT_PROBS,
                                                      max_time_step=self.MAX_TIME_STEP)
        choice_world_generator = ChoiceWorldGenerator(seed=self.SEED,object_color_range=self.OBJECT_COLOR_RANGE)
        multi_world_generator = MultiGenerator(generators=[random_world_generator, choice_world_generator])
        selected_world_generator = multi_world_generator if self.WORLD_GENERATOR == "multi" else (
            random_world_generator if self.WORLD_GENERATOR == "random" else (
                choice_world_generator if self.WORLD_GENERATOR == "choice" else None))
        env = CoopGridWorld(generator=selected_world_generator, compute_reward=self.compute_reward,
                            xenia_lock=self.XENIA_LOCK,xenia_permanence=self.XENIA_PERMANENCE)
        self.env = env
        env.stats.number_communication_channels = self.NUMBER_COMMUNICATION_CHANNELS
        env.stats.size_vocabulary = self.SIZE_VOCABULARY
        env.stats.visible_positions = self.VISIBLE_POSITIONS
        env.stats.recurrency = self.TIME_STEPS
        env.reset()
        self.trainer = Trainer(environment=env, from_save=self.FROM_SAVE, self_play=self.SELF_PLAY, agent_ids=env.stats.agent_ids,
                               state_dim=(env.stats.observation_dimension,), action_dim=env.stats.action_dimension,
                               recurrent=self.RECURRENT,
                               run_name=self.name,
                               max_replay_buffer_size=self.MAX_REPLAY_BUFFER_SIZE,
                               actor_network_generator=partial(create_policy_network, state_dim=env.stats.observation_dimension,
                                                               number_of_big_layers=self.NUMBER_OF_BIG_LAYERS,layer_size=self.LAYER_SIZE, lstm_size=self.LSTM_SIZE, time_steps = self.TIME_STEPS, size_vocabulary = self.SIZE_VOCABULARY, number_communication_channels=self.NUMBER_COMMUNICATION_CHANNELS),
                               critic_network_generator=partial(create_q_network, state_dim=env.stats.observation_dimension,
                                                                action_dim=env.stats.action_dimension, layer_size=self.LAYER_SIZE,lstm_size=self.LSTM_SIZE,
                                                                number_of_big_layers=self.NUMBER_OF_BIG_LAYERS, time_steps=self.TIME_STEPS),
                               seed=self.SEED,
                               env_parallel=self.ENV_PARALLEL, batch_size=self.BATCH_SIZE, learning_rate=self.LEARNING_RATE, alpha=self.ALPHA,
                               target_entropy=self.TARGET_ENTROPY, gamma=self.GAMMA)
        self.trainer.train(training_steps_per_update=self.TRAININGS_PER_TRAINING, render=self.RENDER, pre_sampling_steps=self.PRE_SAMPLING_STEPS, epochs=self.EPOCHS,run_desc=self.name,environment_steps_before_training=self.ENVIRONMENT_STEPS_PER_TRAINING)