import json
import random
from abc import abstractmethod
from functools import partial
from typing import Dict, Any, Callable

import tensorflow as tf
from typing_extensions import Protocol, override

from domain import RENDER, visible_positions_13
from environment.env import CoopGridWorld
from environment.generator import RandomWorldGenerator, ChoiceWorldGenerator, MultiGenerator, ManhattanGenerator
from environment.reward import RaceReward, SingleComputeReward
from training.GenericMLPs1D import create_policy_network, create_critic_network
from training.PPO.PPOTrainer import PPOTrainer
from training.SAC.SACTrainer import SACTrainer
from training.Trainer import Trainer
from utils.loss_logger import LossLogger

"""
Class to ease experiments with multiple configurations. A standard configuration is set by the constants below.
Name and visible_positions are the only parameters that always have to be specified.
For the different algorithms there are different subclasses of this class with constructors suitable for changing
the config.
"""
class Config(Protocol):
    name: str = None
    VISIBLE_POSITIONS: Callable = None
    PLOTTING: bool = True

    # training
    EPOCHS = 2000
    SEED = 18
    ENV_PARALLEL = 32
    FROM_SAVE = False

    # Agent.py
    COM_ALPHA = -0.0
    MOV_ALPHA = 0.18
    GAMMA = 0.99
    TAU = 0.005
    LEARNING_RATE = 0.0001
    RECURRENT = True
    EPSILON = None

    SOCIAL_INFLUENCE_SAMPLE_SIZE = 30
    SOCIAL_REWARD_WEIGHT = 0

    # ExperienceReplayBuffer.py
    BATCH_SIZE = 64
    LSTM_TIME_STEPS = 10
    MAX_REPLAY_BUFFER_SIZE = 10000

    # GenericMLPs1D.py
    LAYER_SIZE = 32
    LSTM_SIZE = 32
    NUMBER_OF_BIG_LAYERS = 2


    # environment
    WORLD_GENERATOR = "random"  # multi or random or choice
    GRID_SIZE_RANGE = (12, 16)
    MAX_TIME_STEP = 30
    NUMBER_OF_AGENTS = 3
    COMMUNISM = False
    AGENT_DROPOUT_PROBS = 0  # 0.5 if NUMBER_OF_AGENTS == 3 else 0  # meaning with 0.5 probabilty the third agent is not placed
    NUMBER_OF_OBJECTS_TO_PLACE_RANGE = (0.2, 0.6)
    OBJECT_COLOR_RANGE = (1, 4)
    POS_REWARD = 1
    NEG_REWARD = -0.05
    CHOICE_NEG_REWARD = 0.
    REWARD_TYPE = "race"
    XENIA_LOCK = True
    XENIA_PERMANENCE = True

    # input
    NUMBER_COMMUNICATION_CHANNELS = 0
    SIZE_VOCABULARY = OBJECT_COLOR_RANGE[1]


    def __call__(self, catched=True):
        if not catched:
            return self._catched_call()
        else:
            try:
                return self._catched_call()
            except Exception as e:
                print(e)
                with open(f"error_{self.name}_error.txt", "w") as f:
                    f.write(str(e))

    def _catched_call(self)->LossLogger:
        random.seed(self.SEED)
        tf.random.set_seed(self.SEED)
        neg_reward = self.NEG_REWARD if self.WORLD_GENERATOR != "choice" else self.CHOICE_NEG_REWARD
        if self.NUMBER_OF_AGENTS == 1:
            self.compute_reward = SingleComputeReward(pos_reward=self.POS_REWARD, neg_reward=neg_reward)
        elif self.REWARD_TYPE == "race":
            self.compute_reward = RaceReward(pos_reward=self.POS_REWARD, neg_reward=neg_reward,
                                             communism=self.COMMUNISM)
        else:
            raise NotImplemented()

        random_world_generator = RandomWorldGenerator(seed=self.SEED, grid_size_range=self.GRID_SIZE_RANGE,
                                                      number_of_agents=self.NUMBER_OF_AGENTS,
                                                      number_of_objects_to_place=self.NUMBER_OF_OBJECTS_TO_PLACE_RANGE,
                                                      number_of_object_colors_range=self.OBJECT_COLOR_RANGE,
                                                      agent_dropout_probs=self.AGENT_DROPOUT_PROBS,
                                                      max_time_step=self.MAX_TIME_STEP)
        choice_world_generator = ChoiceWorldGenerator(seed=self.SEED, object_color_range=self.OBJECT_COLOR_RANGE, number_of_agents=self.NUMBER_OF_AGENTS)
        multi_world_generator = MultiGenerator(generators=[random_world_generator, choice_world_generator])
        selected_world_generator = multi_world_generator if self.WORLD_GENERATOR == "multi" else (
            random_world_generator if self.WORLD_GENERATOR == "random" else (
                choice_world_generator if self.WORLD_GENERATOR == "choice" else None))
        env = CoopGridWorld(generator=selected_world_generator, compute_reward=self.compute_reward,
                            xenia_lock=self.XENIA_LOCK, xenia_permanence=self.XENIA_PERMANENCE)
        self.env = env
        env.stats.number_communication_channels = self.NUMBER_COMMUNICATION_CHANNELS
        env.stats.size_vocabulary = self.SIZE_VOCABULARY
        env.stats.visible_positions = self.VISIBLE_POSITIONS
        env.stats.recurrency = self.LSTM_TIME_STEPS
        env.reset()

        policy_network = partial(create_policy_network, state_dim=env.stats.observation_dimension,
                                 output_activation=self.actor_output_activation,
                                 number_of_big_layers=self.NUMBER_OF_BIG_LAYERS, recurrent=self.RECURRENT,
                                 layer_size=self.LAYER_SIZE, lstm_size=self.LSTM_SIZE,
                                 time_steps=self.LSTM_TIME_STEPS, learning_rate=self.LEARNING_RATE,
                                 size_vocabulary=self.SIZE_VOCABULARY,
                                 number_communication_channels=self.NUMBER_COMMUNICATION_CHANNELS)
        value_network = partial(create_critic_network,
                                state_dim=env.stats.observation_dimension, recurrent=self.RECURRENT,
                                output_dim=self.critic_output_dim(env=env), layer_size=self.LAYER_SIZE,
                                lstm_size=self.LSTM_SIZE, learning_rate=self.LEARNING_RATE,
                                number_of_big_layers=self.NUMBER_OF_BIG_LAYERS,
                                time_steps=self.LSTM_TIME_STEPS, agent_num=self.NUMBER_OF_AGENTS)

        self.trainer = self.get_trainer(env=env, policy_network=policy_network, value_network=value_network)
        self.save()
        tr_result= self.trainer.train(render=RENDER, num_epochs=self.EPOCHS)
        print(tr_result)
        return tr_result

    @abstractmethod
    def save(self):
        ...


    @abstractmethod
    def get_trainer(self, env: CoopGridWorld, policy_network, value_network) -> Trainer:
        ...

    @property
    @abstractmethod
    def actor_output_activation(self)->str:
        ...

    @abstractmethod
    def critic_output_dim(self, env: CoopGridWorld)->int:
        ...

class SACConfig(Config):
    ACTOR_OUTPUT_ACTIVATION = "softmax"
    ENVIRONMENT_STEPS_PER_EPOCH = 500
    BATCHES_PER_EPOCH = 8
    PRE_SAMPLING_STEPS = 1000
    TARGET_ENTROPY = 1.

    def __init__(self, params: Dict[str, Any]):
        for key, value in params.items():
            assert key in self.__dir__(), key
            setattr(self, key, value)

    def get_trainer(self, env: CoopGridWorld, policy_network, value_network):
        return SACTrainer(environment=env, from_save=self.FROM_SAVE,
                          agent_ids=env.stats.agent_ids,
                          state_dim=(env.stats.observation_dimension,), action_dim=env.stats.action_dimension,
                          run_name=self.name,plotting=self.PLOTTING,
                          max_replay_buffer_size=self.MAX_REPLAY_BUFFER_SIZE,
                          social_reward_weight=self.SOCIAL_REWARD_WEIGHT,
                          social_influence_sample_size=self.SOCIAL_INFLUENCE_SAMPLE_SIZE,
                          actor_network_generator=policy_network,
                          critic_network_generator=value_network,
                          seed=self.SEED,
                          env_parallel=self.ENV_PARALLEL, batch_size=self.BATCH_SIZE,
                          mov_alpha=self.MOV_ALPHA,
                          target_entropy=self.TARGET_ENTROPY, gamma=self.GAMMA, com_alpha=self.COM_ALPHA,
                          batches_per_epoch=self.BATCHES_PER_EPOCH, pre_sampling_steps=self.PRE_SAMPLING_STEPS,
                          environment_steps_per_epoch=self.ENVIRONMENT_STEPS_PER_EPOCH, tau=self.TAU, epsilon=self.EPSILON)

    @property
    def actor_output_activation(self) -> str:
        return self.ACTOR_OUTPUT_ACTIVATION

    def critic_output_dim(self, env: CoopGridWorld)->int:
        return env.stats.action_dimension * self.NUMBER_OF_AGENTS

class PPOConfig(Config):
    PPO_EPSILON = 0.2
    GAE_LAMBDA = 0.95
    KLD_THRESHHOLD = 0.05
    STEPS_PER_TRAJECTORIE = 1000
    ACTOR_OUTPUT_ACTIVATION = "log_softmax"
    PREDICT_GOAL_ONLY_AT_END: bool = False

    def __init__(self, params: Dict[str, Any]):
        for key, value in params.items():
            assert key in self.__dir__(), key
            setattr(self, key, value)

    @override
    def save(self):
        attributes = {key: str(getattr(self, key)) for key in self.__dir__() if not key.startswith("_")}
        with open(f"runconfigs/{self.name}.json", "w") as file:
            json.dump(attributes,file)

    @override
    def get_trainer(self, env: CoopGridWorld, policy_network, value_network):
        return PPOTrainer(environment=env, from_save=self.FROM_SAVE,predict_goal_only_at_end=self.PREDICT_GOAL_ONLY_AT_END,
                          agent_ids=env.stats.agent_ids,plotting=self.PLOTTING,
                          state_dim=(env.stats.observation_dimension,), action_dim=env.stats.action_dimension,
                          run_name=self.name,
                          actor_network_generator=policy_network,
                          critic_network_generator=value_network,
                          seed=self.SEED, gae_lambda=self.GAE_LAMBDA,
                          social_influence_sample_size=self.SOCIAL_INFLUENCE_SAMPLE_SIZE,
                          social_reward_weight=self.SOCIAL_REWARD_WEIGHT,
                          env_parallel=self.ENV_PARALLEL, batch_size=self.BATCH_SIZE,
                          alpha=self.MOV_ALPHA,
                          gamma=self.GAMMA, com_alpha=self.COM_ALPHA, epsilon=self.EPSILON,
                          steps_per_trajectory=self.STEPS_PER_TRAJECTORIE, kld_threshold=self.KLD_THRESHHOLD,
                          tau=self.TAU, ppo_epsilon=self.PPO_EPSILON)
    @property
    def actor_output_activation(self) -> str:
        return self.ACTOR_OUTPUT_ACTIVATION

    def critic_output_dim(self, env: CoopGridWorld)->int:
        return self.NUMBER_OF_AGENTS

def make_config(name: str, algo: str, special_vars: Dict[str, Any]):
        params = {"name": name, "VISIBLE_POSITIONS": visible_positions_13}
        params.update(special_vars)
        config_class = SACConfig if algo == "sac" else (PPOConfig if algo == "ppo" else None)
        return config_class(params=params)
