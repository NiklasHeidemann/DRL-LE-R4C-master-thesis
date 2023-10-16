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

runconfigs = [
make_config("", "ppo", {}),
]
for config in runconfigs:
    config(catched=False)
