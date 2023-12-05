import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from bayes_opt import BayesianOptimization

from experiments import medium_choice, easy_choice
from plotting.plots import plot_multiple
from runconfig import make_config
from utils.loss_logger import LossLogger

"""
File covering the hyperparameter optimization. For usage, see commented lines at the very bottom of this file.
"""

current_experiment = "hpo_exp_3"

"""
Function that loads previous runs and registers them with the optimizer. A run consists of a json with the parameter
configuration and the logged values from which the run-value is calculated.
"""
def register_previous(optimizer: BayesianOptimization):
    path = Path("logger/")
    all = defaultdict(list)
    for file in os.listdir(path):
        if not file.startswith(current_experiment) or file.endswith(".json"): # if not from curr
            continue
        logger = LossLogger(file.rsplit("_logger",maxsplit=1)[0])
        logger.load(path=str(path))
        result = logger.avg_last(identifier="test_return", n=20)

        file_parts = file.split("_")
        params = {}
        params["com_alpha"] = float(file_parts[3][2:])
        params["mov_alpha"] = float(file_parts[4][2:])
        params["learning_rate_exp"] = float(file_parts[5][2:])
        params["batch_size_exp"] = float(file_parts[6][2:])
        params["layer_size"] = float(file_parts[7][2:])
        params["gamma_inverse_exp"] = float(file_parts[8][2:])
        all[str(params)].append(result)

        if len(all[str(params)])==3:
            optimizer.register(params=params, target=np.median(all[str(params)]))
    for key in all:
        all[key] = np.median(all[key])
    # print so that it can be copied as python code
    print("all = {")
    for key in all:
        print(f"{key}: {all[key]},")
    print("}")
    return optimizer

"""
Function evaluating a given parameter configuration. Calls three times the same configuration with different seeds and returns the median of the three results.
The result is the average test return of the last 200 episodes. 
"""
def wrapper_exp_2(com_alpha: float, mov_alpha:float, learning_rate_exp: float, batch_size_exp: int, layer_size: int, gamma_inverse_exp: float):
    # define the experiment name
    name = f"{current_experiment}_ca{com_alpha}_ma{mov_alpha}_lr{learning_rate_exp}_bs{batch_size_exp}_ls{layer_size}_gi{gamma_inverse_exp}"
    seeds = [19,20,21]
    configs = [make_config(name=f"{name}_{seed}", algo="ppo", special_vars={"EPOCHS": 3000, "SEED":seed, "PLOTTING": False,"FROM_SAVE":False, "PREDICT_GOAL_ONLY_AT_END":True,
                                                                "COM_ALPHA":com_alpha, **easy_choice,
                                                                "LEARNING_RATE":10**learning_rate_exp, "BATCH_SIZE":int(2**batch_size_exp),
                                                                "MOV_ALPHA":mov_alpha, "LAYER_SIZE":int(layer_size),
                                                                "GAMMA":1-10**gamma_inverse_exp
                                                                }) for seed in seeds]
    avg_returns = []
    for config in configs:
        try:
            loss_logger = config(catched=False)
            avg_returns.append(loss_logger.avg_last(identifier="test_return", n=20))# avg of last 20 test returns from the last 200 episodes
        except Exception as e:
            print("Exception, aborting run with -1000", e)
            return -1000
        print(name, "done with", avg_returns[-1])
    return np.median(avg_returns)

optimizer = BayesianOptimization(
    f=wrapper_exp_2,
    pbounds={
        # parameter space. Note that these are linear scales and some of those will be transformed by the wrapper.
        "learning_rate_exp": (-5., -1),
        "batch_size_exp": (2., 8.),
        "gamma_inverse_exp": (-4.,-1),
        "com_alpha": (0.,0.2),
        "mov_alpha": (0.,1.),
        "layer_size": (16, 256)
    },
    verbose=2,
    random_state=1,
)

# uncomment the following line if you want to include previous runs
# register_previous(optimizer=optimizer)

# use lines like this if you want to test specific configurations
#optimizer.probe(params={'batch_size_exp': 4.728153365314442, 'com_alpha': 0.11757430125923729, 'gamma_inverse_exp': -3.207587067368291, 'layer_size': 89.269191843384, 'learning_rate_exp': -3.51338169310327, 'mov_alpha': 0.24444799937161088}, lazy=False)

# uncomment the following line if you want to run the optimization
#optimizer.maximize(init_points=0,    n_iter=0,)