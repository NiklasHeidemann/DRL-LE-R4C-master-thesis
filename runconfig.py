from abc import abstractmethod
from functools import partial
from typing import Dict, Any, Callable

from typing_extensions import Protocol

import params
from PPO.PPOTrainer import PPOTrainer
from training.GenericMLPs1D import create_policy_network, create_critic_network
from SAC.SACTrainer import SACTrainer
from environment.env import CoopGridWorld
from environment.generator import RandomWorldGenerator, ChoiceWorldGenerator, MultiGenerator
from environment.reward import RaceReward, SingleComputeReward
import random
import tensorflow as tf

class Config(Protocol):
    name: str = None
    VISIBLE_POSITIONS: Callable = None
    # network
    BATCH_SIZE = 64
    LAYER_SIZE = 32
    LEARNING_RATE = 0.0001
    LSTM_SIZE = 32
    NUMBER_OF_BIG_LAYERS = 2
    RECURRENT = True
    SELF_PLAY = True
    TIME_STEPS = 10
    ALPHA = 0.2
    L_ALPHA = 0.01
    GAMMA = 0.99
    TARGET_ENTROPY = 1.

    #socialinfluence
    SOCIAL_INFLUENCE_SAMPLE_SIZE = 30
    SOCIAL_REWARD_WEIGHT = 0

    # environment
    WORLD_GENERATOR = "random"  # multi or random or choice
    GRID_SIZE_RANGE = (4, 16)
    MAX_TIME_STEP = 30
    NUMBER_OF_AGENTS = 3
    COMMUNISM = False
    AGENT_DROPOUT_PROBS = 0#0.5 if NUMBER_OF_AGENTS == 3 else 0  # meaning with 0.5 probabilty the third agent is not placed
    NUMBER_OF_OBJECTS_TO_PLACE_RANGE = (0.2, 0.6)
    OBJECT_COLOR_RANGE = (10, 20)
    POS_REWARD = 2
    NEG_REWARD = -0.1
    REWARD_TYPE = "race"
    XENIA_LOCK = False
    XENIA_PERMANENCE = True

    # input
    NUMBER_COMMUNICATION_CHANNELS = 1
    SIZE_VOCABULARY = OBJECT_COLOR_RANGE[1]

    # buffer
    MAX_REPLAY_BUFFER_SIZE = 10000
    PRE_SAMPLING_STEPS = 100

    # training
    ENVIRONMENT_STEPS_PER_TRAINING = 500
    TRAININGS_PER_TRAINING = 8
    EPOCHS = 200000
    SEED = 15
    ENV_PARALLEL = 32
    FROM_SAVE = True
    RENDER = params.RENDER

    def __call__(self, catched=True):
        if not catched:
            self._catched_call()
        else:
            try:
                self._catched_call()
            except Exception as e:
                print(e)
                with open(f"error_{self.name}_error.txt", "w") as f:
                    f.write(str(e))


    def _catched_call(self):
        random.seed(self.SEED)
        if self.NUMBER_OF_AGENTS == 1:
            self.compute_reward = SingleComputeReward(pos_reward=self.POS_REWARD, neg_reward=self.NEG_REWARD)
        elif self.REWARD_TYPE == "race":
            self.compute_reward = RaceReward(pos_reward=self.POS_REWARD, neg_reward=self.NEG_REWARD, communism=self.COMMUNISM)
        else:
            raise NotImplemented()
        tf.random.set_seed(self.SEED)

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
        self.trainer = self.get_trainer(env=env)
        self.trainer.train(training_steps_per_update=self.TRAININGS_PER_TRAINING, render=self.RENDER,
                           pre_sampling_steps=self.PRE_SAMPLING_STEPS, epochs=self.EPOCHS, run_desc=self.name,
                           environment_steps_before_training=self.ENVIRONMENT_STEPS_PER_TRAINING)


    @abstractmethod
    def get_trainer(self, env):
        ...

class SACConfig(Config):
    ACTOR_OUTPUT_ACTIVATION = "softmax"
    def __init__(self, params: Dict[str, Any]):
        for key, value in params.items():
            assert key in self.__dir__()
            setattr(self, key, value)

    def get_trainer(self, env):
        output_dim = env.stats.action_dimension * self.NUMBER_OF_AGENTS
        return SACTrainer(environment=env, from_save=self.FROM_SAVE, self_play=self.SELF_PLAY, agent_ids=env.stats.agent_ids,
                          state_dim=(env.stats.observation_dimension,), action_dim=env.stats.action_dimension,
                          recurrent=self.RECURRENT,
                          run_name=self.name,
                          max_replay_buffer_size=self.MAX_REPLAY_BUFFER_SIZE,
                          actor_network_generator=partial(create_policy_network, state_dim=env.stats.observation_dimension, output_activation=self.ACTOR_OUTPUT_ACTIVATION,
                                                               number_of_big_layers=self.NUMBER_OF_BIG_LAYERS,layer_size=self.LAYER_SIZE, lstm_size=self.LSTM_SIZE, time_steps = self.TIME_STEPS, size_vocabulary = self.SIZE_VOCABULARY, number_communication_channels=self.NUMBER_COMMUNICATION_CHANNELS),
                          critic_network_generator=partial(create_critic_network, state_dim=env.stats.observation_dimension,
                                                                output_dim=output_dim, layer_size=self.LAYER_SIZE,lstm_size=self.LSTM_SIZE,
                                                                number_of_big_layers=self.NUMBER_OF_BIG_LAYERS, time_steps=self.TIME_STEPS),
                          seed=self.SEED,
                          env_parallel=self.ENV_PARALLEL, batch_size=self.BATCH_SIZE, learning_rate=self.LEARNING_RATE, alpha=self.ALPHA,
                          target_entropy=self.TARGET_ENTROPY, gamma=self.GAMMA, l_alpha=self.L_ALPHA)

class PPOConfig(Config):
    EPSILON = 0.2
    GAE_LAMBDA = 0.95
    KLD_THRESHHOLD = 0.05
    STEPS_PER_TRAJECTORIE = 1000
    ACTOR_OUTPUT_ACTIVATION = "log_softmax"

    def __init__(self, params: Dict[str, Any]):
        for key, value in params.items():
            assert key in self.__dir__()
            setattr(self, key, value)

    def get_trainer(self, env):
        return PPOTrainer(environment=env, from_save=self.FROM_SAVE, self_play=self.SELF_PLAY, agent_ids=env.stats.agent_ids,
                               state_dim=(env.stats.observation_dimension,), action_dim=env.stats.action_dimension,
                               recurrent=self.RECURRENT,
                               run_name=self.name,
                               max_replay_buffer_size=self.MAX_REPLAY_BUFFER_SIZE,
                               actor_network_generator=partial(create_policy_network, state_dim=env.stats.observation_dimension,output_activation = self.ACTOR_OUTPUT_ACTIVATION,
                                                               number_of_big_layers=self.NUMBER_OF_BIG_LAYERS,layer_size=self.LAYER_SIZE, lstm_size=self.LSTM_SIZE, time_steps = self.TIME_STEPS, size_vocabulary = self.SIZE_VOCABULARY, number_communication_channels=self.NUMBER_COMMUNICATION_CHANNELS),
                               critic_network_generator=partial(create_critic_network, state_dim=env.stats.observation_dimension,
                                                                output_dim=self.NUMBER_OF_AGENTS, layer_size=self.LAYER_SIZE,lstm_size=self.LSTM_SIZE,
                                                                number_of_big_layers=self.NUMBER_OF_BIG_LAYERS, time_steps=self.TIME_STEPS),
                               seed=self.SEED, gae_lambda=self.GAE_LAMBDA,social_influence_sample_size=self.SOCIAL_INFLUENCE_SAMPLE_SIZE,social_reward_weight=self.SOCIAL_REWARD_WEIGHT,
                               env_parallel=self.ENV_PARALLEL, batch_size=self.BATCH_SIZE, learning_rate=self.LEARNING_RATE, alpha=self.ALPHA,
                               target_entropy=self.TARGET_ENTROPY, gamma=self.GAMMA, l_alpha=self.L_ALPHA, epsilon=self.EPSILON, steps_per_trajectory=self.STEPS_PER_TRAJECTORIE,kld_threshold=self.KLD_THRESHHOLD)
