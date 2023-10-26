from collections import defaultdict
from typing import List, Any, Optional, Dict

import numpy as np

ACTOR_LOSS = "actor_losses"
OTHER_ACTOR_LOSS = "other_actor_losses"
CRITIC_LOSS = "critic_losses"
KLD = "kld"
SOCIAL_REWARD = "social_reward"
LOG_PROBS = "log_probs"
RETURNS = "returns"
Q_VALUES = "q_values"
V_VALUES = "v_values"
MAX_Q_VALUES = "max_q_values"
ALPHA_VALUES = "alpha"
ENTROPY = "mov_entropy"
COM_ENTROPY = "l_entropy"
AVG_ADVANTAGE = "avg_advantage"
STD_ADVANTAGE = "std_advantage"
N_AGENT_RETURNS = lambda n : f"{n}_agent_returns"
TEST_RETURNS = "test_return"
TEST_SOCIAL_RETURNS = "test_social_return"
PREDICTED_GOAL = "predicted_goal"

class LossLogger:

    def __init__(self):
        self._var_dict: Dict[str, List[Any]] = {}
        self._is_smoothed: Dict[str, Optional[int]] = {}
        self._smoothed: Dict[str, List[Any]] = {}
        self._aggregator: Dict[str, List[float]] = defaultdict(list)

    def add_list(self, identifier: str, smoothed: Optional[int]=None)->None:
        self._var_dict[identifier] = []
        self._is_smoothed[identifier] = smoothed
        if smoothed is not None:
            self._smoothed[identifier] = []

    def add_lists(self, identifiers: List[str], smoothed: Optional[int]=None)->None:
        for identifier in identifiers:
            self.add_list(identifier=identifier, smoothed=smoothed)

    def add_value(self, identifier: str, value: Any)->None:
        self._var_dict[identifier].append(value)
        if self._is_smoothed[identifier] is not None:
            self._smoothed[identifier].append(np.average(self.avg_last(identifier=identifier,n=self._is_smoothed[identifier])))

    def add_values(self, dict_: Dict[str, Any])->None:
        for identifier, value in dict_.items():
            self.add_value(identifier=identifier,value=value)

    def add_value_list(self, identifier: str, values: List[Any])->None:
        for value in values:
            self.add_value(identifier=identifier,value=value)

    def add_aggregatable_value(self, identifier: str, value: float)->None:
        self._aggregator[identifier].append(value)

    def add_aggregatable_values(self, dict_: Dict[str, Any])->None:
        for identifier, value in dict_.items():
            self.add_aggregatable_value(identifier=identifier,value=value)


    def avg_aggregatable(self, identifier: str)->None:
        self.add_value(identifier=identifier,value=np.average(self._aggregator[identifier]))
        self._aggregator[identifier] = []

    def avg_aggregatables(self, identifiers: List[str])->None:
        for identifier in identifiers:
            self.avg_aggregatable(identifier=identifier)

    def last(self, identifier: str)->Any:
        return self._var_dict[identifier][-1] if len(self._var_dict[identifier])>0 else float("nan")

    def avg_last(self, identifier:str, n: int)->float:
        return np.average(self._var_dict[identifier][-n:])

    def all_smoothed(self)->Dict[str,List[float]]:
        return self._smoothed

    def save(self, path: str)->None:
        np.savez(f"{path}/logger_var_dict", **self._var_dict)
        np.savez(f"{path}/logger_smoothed", **self._smoothed)
        np.savez(f"{path}/logger_is_smoothed", **self._is_smoothed)

    def load(self, path: str)->None:
        self._var_dict = {key: list(value) for key, value in dict(np.load(f"{path}/logger_var_dict.npz")).items()}
        self._smoothed = {key: list(value) for key, value in dict(np.load(f"{path}/logger_smoothed.npz")).items()}
        self._is_smoothed = {key: value for key, value in dict(np.load(f"{path}/logger_is_smoothed.npz")).items()}