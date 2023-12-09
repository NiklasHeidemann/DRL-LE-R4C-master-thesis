# Overview

- For access to the master thesis pdf, please contact me or the resarch group supervisor.

- for how to run experiments, look at main.py and experiments.py. All configurable parameters can be found in runconfig.py 
- for running a hyperparameter search, look and execute hpo.py

- for visualization of the test episodes, change the render parameter in domain.py

## Experiment data
- the data of the experiment runs can be found in the folder 'experiments'. All experiments are named. Each folder contains the json-files with the run configurations, the metric plots, the saved tensorflow models and the logged metrics. The experiments are:
  - LE: 4 seeds in the choice environment with 2 agents. One seed exhibited language
  - oR4C: 4 seeds in the choice environment with 3 agents. No seed exhibited language
  - R4C_easy: 2 agents vs 3 cooperating agents vs 3 R4C-agents x communication vs no communication in the easy random env. No run exhibited language, environment is too easy.
  - R4C_nocom: Nocom 3 cooperating agents in harder random environments. Established baselines
  - R4C_medium: 3 cooperating agents vs 3 R4C-agents in the medium random env, 2 seeds each. No run exhibited language, LE is too infrequent.
  - SAC: Attempt to compare SAC with PPO. Unoptimized SAC is too bad to do this systematically.
  - HPO: Attempt to optimize PPO parameters, with the available computational resources, no improvement was found, especially no LE.
  - SocialReward: Experiment with adding social influence as reward (see Jaques et al). Only detrimental effects found.


## Code structure
- top level files are executable scripts (main, hpo) or configuration utils (runconfig, experiments, domain)
- environment contains:
  - generator.py: three different generators that create grid world instances
  - env.py: The actual environment, that uses a generator for each episode.
  - render.py: Visualization, use it by changing the render parameter in domain.py
  - reward.py: Different reward definitions
  - stats.py: Saving all parameters of a grid world instance
- experiments (see above)
- language: Files for the computation of the two metrics:
    - predictGoal: positive signaling/ProbeClassifiers
    - SocialRewardComputer: positive listening/Social Influence
- plotting: Utils for visualizing the metrics, only plots.py is used by the rest of the code, the rest are separate scripts
- Training: Split in general (often abstract) training classes and the specific parts of PPO and SAC (use of PPO is very much recommended)
- utils: Metric logging and a helper for code execution time stopping