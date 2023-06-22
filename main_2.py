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
    #make_config("a_three_race_com", {"COMMUNISM":False, "NUMBER_OF_AGENTS":3, "NUMBER_COMMUNICATION_CHANNELS": 1,"AGENT_DROPOUT_PROBS":0., "XENIA_LOCK": False, "ALPHA": 0.06}),
    make_config("b_three_race_com", {"COMMUNISM":False, "NUMBER_OF_AGENTS":3, "NUMBER_COMMUNICATION_CHANNELS": 1,"AGENT_DROPOUT_PROBS":0., "XENIA_LOCK": False, "ALPHA": 0.08, "L_ALPHA": 0.02}),
    #make_config("c_three_race_com", {"COMMUNISM":False, "NUMBER_OF_AGENTS":3, "NUMBER_COMMUNICATION_CHANNELS": 1,"AGENT_DROPOUT_PROBS":0., "XENIA_LOCK": False, "ALPHA": 0.03}),
    #make_config("d_three_race_no_com", {"COMMUNISM":False, "NUMBER_OF_AGENTS":3, "NUMBER_COMMUNICATION_CHANNELS": 0,"AGENT_DROPOUT_PROBS":0., "XENIA_LOCK": False}),
    #make_config("three_coop_com", {"COMMUNISM":True, "NUMBER_OF_AGENTS":3, "NUMBER_COMMUNICATION_CHANNELS": 1,"AGENT_DROPOUT_PROBS":0.}),
    #make_config("three_coop_no_com", {"COMMUNISM":True, "NUMBER_OF_AGENTS":3, "NUMBER_COMMUNICATION_CHANNELS": 0,"AGENT_DROPOUT_PROBS":0.}),
]
for config in runconfigs:
    config()
#sac_agent._agent._critic_1.summary()
#sac_agent._agent._actor.summary() if SELF_PLAY else sac_agent._agent._actors["0"].summary()
#sac_agent.test(n_samples=2, verbose_samples=0)
#sac_agent.train(epochs=EPOCHS, pre_sampling_steps=PRE_SAMPLING_STEPS, environment_steps_before_training=ENVIRONMENT_STEPS_PER_TRAINING, render=RENDER, training_steps_per_update=TRAININGS_PER_TRAINING)
#sac_agent.test(n_samples=20, verbose_samples=0)
#TrainPredictGoal()(environment=rc.env,agent=rc.trainer._agent)
