# Overview

- For access to the master thesis pdf, please contact me or the resarch group supervisor.
- the data of the run experiments can be found in the folder experiments. All experiments are named. Each folder contains the json-files with the run configurations, the metric plots, the saved tensorflow models and the logged metrics. The experiments are:
  - LE: 4 seeds in the choice environment with 2 agents. One seed exhibited language
  - oR4C: 4 seeds in the choice environment with 3 agents. No seed exhibited language
  - R4C_easy: 2 agents vs 3 cooperating agents vs 3 R4C-agents x communication vs no communication in the easy random env. No run exhibited language, environment is too easy.
  - R4C_nocom: Nocom 3 cooperating agents in harder random environments. Established baselines
  - R4C_medium: 3 cooperating agents vs 3 R4C-agents in the medium random env. No run exhibited language, LE is too infrequent.
  - SAC: Attempt to compare SAC with PPO. Unoptimized SAC is too bad.
  - HPO: Attempt to optimize PPO parameters, with the available computational resources, no improvement was found, especially no LE.
  - SocialReward: Experiment with adding social influence as reward (see Jaques et al). Only detrimental effects found.

- for how to run experiments, look at main.py and experiments.py. All configurable parameters can be found in runconfig.py 
- for running a hyperparameter search, look and execute hpo.py

- for visualization of the test episodes, change the render parameter in domain.py