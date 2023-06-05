from collections import defaultdict
from typing import List, Any, Optional, Dict
import numpy as np

ACTOR_LOSS = "actor_losses"
OTHER_ACTOR_LOSS = "other_actor_losses"
CRITIC_LOSS = "critic_losses"
LOG_PROBS = "log_probs"
RETURNS = "returns"
Q_VALUES = "q_values"
MAX_Q_VALUES = "max_q_values"
ALPHA_VALUES = "alpha"
ENTROPY = "entropy"

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

    def load(self, path: str)->None:
        self._var_dict = dict(np.load(f"{path}/logger_var_dict.npz"))
        self._smoothed = dict(np.load(f"{path}/logger_smoothed.npz"))