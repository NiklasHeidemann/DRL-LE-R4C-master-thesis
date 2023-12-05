from typing import Any, Dict

import matplotlib

from domain import visible_positions_13
from experiments import exp_9
from runconfig import make_config

matplotlib.use("agg")

"""
# example usage. Define a list of configurations with the make_config util and the call them in a for loop:
runconfigs = [
    make_config("no_com_coop", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 0}),
    make_config("no_com_race", "ppo", {"COMMUNISM":False, "NUMBER_COMMUNICATION_CHANNELS": 0,}),
    make_config("com_race", "ppo", {"COMMUNISM":False, "NUMBER_COMMUNICATION_CHANNELS": 2,}),
    make_config("com_coop", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 2,}),
    make_config("2_no_com", "ppo", {"COMMUNISM":False, "NUMBER_COMMUNICATION_CHANNELS": 0, "NUMBER_OF_AGENTS": 2}),
    make_config("2_com", "ppo", {"COMMUNISM":False, "NUMBER_COMMUNICATION_CHANNELS": 2, "NUMBER_OF_AGENTS": 2}),
]
"""

runconfigs = exp_9
for config in runconfigs:
    config(catched=False)
