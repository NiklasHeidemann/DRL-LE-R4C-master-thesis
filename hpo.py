import numpy as np
from bayes_opt import BayesianOptimization

from experiments import medium_choice
from runconfig import make_config


def wrapper_exp_2(com_alpha: float, mov_alpha:float, learning_rate_exp: float, batch_size_exp: int, layer_size: int, gamma_inverse_exp: float):
    name = f"hpo_exp_2_ca{com_alpha}_ma{mov_alpha}_lr{learning_rate_exp}_bs{batch_size_exp}_ls{layer_size}_gi{gamma_inverse_exp}"
    seeds = [19,20,21]
    configs = [make_config(name=f"{name}_{seed}", algo="ppo", special_vars={"EPOCHS": 2000, "SEED":seed, "PLOTTING": False, "PREDICT_GOAL_ONLY_AT_END":True,
                                                                "COM_ALPHA":com_alpha, **medium_choice,
                                                                "LEARNING_RATE":10**learning_rate_exp, "BATCH_SIZE":int(2**batch_size_exp),
                                                                "MOV_ALPHA":mov_alpha, "LAYER_SIZE":layer_size,
                                                                "GAMMA":1-10**gamma_inverse_exp
                                                                }) for seed in seeds]
    avg_returns = []
    for config in configs:
        loss_logger = config()
        avg_returns.append(loss_logger.avg_last(identifier="test_return", n=20))
        print(name, "done with", avg_returns[-1])
    return np.median(avg_returns)

optimizer = BayesianOptimization(
    f=wrapper_exp_2,
    pbounds={
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
optimizer.maximize(
    init_points=2,
    n_iter=8,
)