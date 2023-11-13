# environments
from runconfig import make_config


def choice_env(num_colors: int, num_agents: int):
 return {
    "WORLD_GENERATOR": "choice",
    "OBJECT_COLOR_RANGE": (None, num_colors),
     "NUMBER_OF_AGENTS": num_agents,
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

thompson_sampling = {"EPSILON": None, "COM_ALPHA": 0.01}
epsilon_sampling = {"EPSILON": 0.1, "COM_ALPHA": 0.}

medium_choice = {**choice_env(8,2),"NUMBER_COMMUNICATION_CHANNELS": 2}
easy_choice = {**choice_env(4,2),"NUMBER_COMMUNICATION_CHANNELS": 1}

exp_1_le_in_choice = [
    make_config("exp_1_a_no_com_easy_eps", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 0, **choice_env(4,2),**epsilon_sampling, "EPOCHS": 1000}),
    #make_config("exp_1_b_com_easy_eps", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2),**epsilon_sampling}),
    #make_config("exp_1_c_com_hard_eps", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(12,2),**epsilon_sampling}),
    #make_config("exp_1_d_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2),**thompson_sampling}),
    #make_config("exp_1_e_com_hard_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(12,2),**thompson_sampling}),
    #make_config("exp_1_f_no_com_hard_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 0, **choice_env(12,2),**thompson_sampling, "EPOCHS": 1000})
    #make_config("exp_1_g_com_medium_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(8,2),**thompson_sampling, "EPOCHS": 3000}),
    #make_config("exp_1_h_com_easy_eps", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2),"EPSILON": 0.05, "COM_ALPHA": 0.}),
]
exp_2_le_in_random = [
    make_config("exp_2_a_no_com_very_easy", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,  **random_env_by_difficulty(communism=True, number_of_agents=2, difficulty="very easy")}),
    make_config("exp_2_b_com_very_easy", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,  **random_env_by_difficulty(communism=True, number_of_agents=2, difficulty="very easy")}),
    make_config("exp_2_c_no_com_easy", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,  **random_env_by_difficulty(communism=True, number_of_agents=2, difficulty="easy")}),
    make_config("exp_2_d_com_easy", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,  **random_env_by_difficulty(communism=True, number_of_agents=2, difficulty="easy")}),
    make_config("exp_2_e_no_com_medium", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,  **random_env_by_difficulty(communism=True, number_of_agents=2, difficulty="medium")}),
    make_config("exp_2_f_com_medium", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,  **random_env_by_difficulty(communism=True, number_of_agents=2, difficulty="medium")}),
    make_config("exp_2_g_no_com_hard", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,  **random_env_by_difficulty(communism=True, number_of_agents=2, difficulty="hard")}),
    make_config("exp_2_h_com_hard", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,  **random_env_by_difficulty(communism=True, number_of_agents=2, difficulty="hard")}),
    ]