import random
from typing import Any, Dict

import matplotlib
import tensorflow as tf

from domain import visible_positions_13
from environment.reward import RaceReward
from language.predictGoal import TrainPredictGoal
from runconfig import RunConfig

matplotlib.use("agg")

def make_config(name: str, special_vars: Dict[str, Any]):
    params = {"name": name, "VISIBLE_POSITIONS": visible_positions_13}
    params.update(special_vars)
    return RunConfig(params=params)

runconfigs = [
    make_config("with_race", {"COMMUNISM":False, "NUMBER_OF_AGENTS":3}),
    make_config("coop_race", {"COMMUNISM":True, "NUMBER_OF_AGENTS":3}),
    make_config("no_race", {"COMMUNISM":False, "NUMBER_OF_AGENTS":2}),
]
for config in runconfigs:
    config()
#sac_agent._agent._critic_1.summary()
#sac_agent._agent._actor.summary() if SELF_PLAY else sac_agent._agent._actors["0"].summary()
#sac_agent.test(n_samples=2, verbose_samples=0)
#sac_agent.train(epochs=EPOCHS, pre_sampling_steps=PRE_SAMPLING_STEPS, environment_steps_before_training=ENVIRONMENT_STEPS_PER_TRAINING, render=RENDER, training_steps_per_update=TRAININGS_PER_TRAINING)
#sac_agent.test(n_samples=20, verbose_samples=0)
#TrainPredictGoal()(environment=rc.env,agent=rc.trainer._agent)
