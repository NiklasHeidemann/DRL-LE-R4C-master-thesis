from typing import Any, Dict

import matplotlib

from domain import visible_positions_13
from runconfig import SACConfig, PPOConfig

matplotlib.use("agg")

def make_config(name: str, algo:str, special_vars: Dict[str, Any]):
    params = {"name": name, "VISIBLE_POSITIONS": visible_positions_13}
    params.update(special_vars)
    config_class = SACConfig if algo == "sac" else (PPOConfig if algo == "ppo" else None)
    return config_class(params=params)

#group_image()
runconfigs = [
    make_config("no_com_coop", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 0}),
    make_config("no_com_race", "ppo", {"COMMUNISM":False, "NUMBER_COMMUNICATION_CHANNELS": 0,}),
    make_config("com_race", "ppo", {"COMMUNISM":False, "NUMBER_COMMUNICATION_CHANNELS": 2,}),
    make_config("com_coop", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 2,}),
    make_config("2_no_com", "ppo", {"COMMUNISM":False, "NUMBER_COMMUNICATION_CHANNELS": 0, "NUMBER_OF_AGENTS": 2}),
    make_config("2_com", "ppo", {"COMMUNISM":False, "NUMBER_COMMUNICATION_CHANNELS": 2, "NUMBER_OF_AGENTS": 2}),
]

runconfigs = [
make_config("com", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 2}),
make_config("no_com", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0}),
]
for config in runconfigs:
    config(catched=False)
