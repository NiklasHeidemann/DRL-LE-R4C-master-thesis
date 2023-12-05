from runconfig import make_config

"""
File containing utils for running experiments. Groups of parameters that are reused in multiple experiments are defined 
here. One can also define a list of training runs, which can then be executed at once.
For the experiments that have been performed for the thesis, see the jsons in the experiments folder.
"""

def choice_env(num_colors: int, num_agents: int, communism: bool):
 return {
    "WORLD_GENERATOR": "choice",
    "OBJECT_COLOR_RANGE": (None, num_colors),
     "NUMBER_OF_AGENTS": num_agents,
     "COMMUNISM"      : communism,
 }
def random_env(num_colors: [int,int], xenia_lock: bool, xenia_permanence: bool, communism: bool, number_of_agents: int, grid_size_range: [int,int]):
 return {
    "WORLD_GENERATOR": "random",
    "OBJECT_COLOR_RANGE": num_colors,
     "XENIA_LOCK": xenia_lock,
     "XENIA_PERMANENCE": xenia_permanence,
     "COMMUNISM": communism,
     "NUMBER_OF_AGENTS": number_of_agents,
     "GRID_SIZE_RANGE": grid_size_range,
 }

def random_env_by_difficulty(communism: bool, number_of_agents: int, difficulty: str):
    if difficulty=="very easy":
        return random_env(num_colors=[1,4], xenia_lock=True, xenia_permanence=False, communism=communism, number_of_agents=number_of_agents, grid_size_range=[4,10])
    elif difficulty=="easy":
        return random_env(num_colors=[10,20], xenia_lock=True, xenia_permanence=False, communism=communism, number_of_agents=number_of_agents, grid_size_range=[12,16])
    elif difficulty=="medium":
        return random_env(num_colors=[10,20], xenia_lock=False, xenia_permanence=False, communism=communism, number_of_agents=number_of_agents, grid_size_range=[12,16])
    elif difficulty=="hard":
        return random_env(num_colors=[10,20], xenia_lock=True, xenia_permanence=True, communism=communism, number_of_agents=number_of_agents, grid_size_range=[12,16])

thompson_sampling = {"EPSILON": None, "COM_ALPHA": 0.001}
epsilon_sampling = {"EPSILON": 0.1, "COM_ALPHA": 0.}

medium_choice = {**choice_env(8,2, True),"NUMBER_COMMUNICATION_CHANNELS": 2}
easy_choice = {**choice_env(4,2, True),"NUMBER_COMMUNICATION_CHANNELS": 1}
exp_9 = [
make_config("exp_9_a_com_easy_thomp", "ppo", {"SOCIAL_REWARD_WEIGHT":0.01,"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2,False),**thompson_sampling, "EPOCHS": 6000, "SEED":30}),
make_config("exp_9_b_com_easy_thomp", "ppo", {"SOCIAL_REWARD_WEIGHT":0.05,"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2,False),**thompson_sampling, "EPOCHS": 6000, "SEED":31}),
make_config("exp_9_c_com_easy_thomp", "ppo", {"SOCIAL_REWARD_WEIGHT":0.1,"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2,False),**thompson_sampling, "EPOCHS": 6000, "SEED":32}),
make_config("exp_9_d_com_easy_thomp", "ppo", {"SOCIAL_REWARD_WEIGHT":1,"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2,False),**thompson_sampling, "EPOCHS": 6000, "SEED":33}),
]