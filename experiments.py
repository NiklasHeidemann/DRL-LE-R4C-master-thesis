# environments
from runconfig import make_config


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
exp_6 = [
make_config("exp_9_a_com_easy_thomp", "ppo", {"SOCIAL_REWARD_WEIGHT":0.01,"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2,False),**thompson_sampling, "EPOCHS": 6000, "SEED":30}),
make_config("exp_9_b_com_easy_thomp", "ppo", {"SOCIAL_REWARD_WEIGHT":0.05,"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2,False),**thompson_sampling, "EPOCHS": 6000, "SEED":31}),
make_config("exp_9_c_com_easy_thomp", "ppo", {"SOCIAL_REWARD_WEIGHT":0.1,"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2,False),**thompson_sampling, "EPOCHS": 6000, "SEED":32}),
make_config("exp_9_d_com_easy_thomp", "ppo", {"SOCIAL_REWARD_WEIGHT":1,"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2,False),**thompson_sampling, "EPOCHS": 6000, "SEED":33}),
#make_config("sac_choice", "sac", {"FROM_SAVE":False,"NUMBER_COMMUNICATION_CHANNELS": 1,**choice_env(4,2,True),**thompson_sampling, "EPOCHS": 100000, "SEED":30}),
#make_config("sac_exp_6_b_no_com_r_easy_thomp", "sac", {"FROM_SAVE":True,"NUMBER_COMMUNICATION_CHANNELS": 0,**random_env_by_difficulty(True,2,"very easy"),**thompson_sampling, "EPOCHS": 10000000, "SEED":30}),
    #make_config("exp_8_a_com_3c_middle", "ppo", {"FROM_SAVE":True,"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(False,3,"easy"),**thompson_sampling, "EPOCHS": 4501, "SEED":30}),
#make_config("exp_8_b_com_3c_middle", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(True,3,"easy"),**thompson_sampling, "EPOCHS": 4001, "SEED":31}),
#make_config("exp_8_c_com_3v_middle", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(False,3,"easy"),**thompson_sampling, "EPOCHS": 4001, "SEED":32}),
#make_config("exp_8_d_com_3v_middle", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(False,3,"easy"),**thompson_sampling, "EPOCHS": 4001, "SEED":33}),
]
_exp_6 = [
    make_config("exp_6_a_com_r_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(True,2,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
    make_config("exp_6_b_no_com_r_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,**random_env_by_difficulty(True,2,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
    make_config("exp_6_c_com_r_3c_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(True,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
    make_config("exp_6_d_no_com_r_3c_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,**random_env_by_difficulty(True,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
    make_config("exp_6_e_com_r_3v_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(False,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
    make_config("exp_6_f_no_com_r_3v_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,**random_env_by_difficulty(False,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
    make_config("exp_6_a2_com_r_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(True,2,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":31}),
    make_config("exp_6_b2_no_com_r_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,**random_env_by_difficulty(True,2,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":31}),
    make_config("exp_6_c2_com_r_3c_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(True,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":31}),
    make_config("exp_6_d2_no_com_r_3c_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,**random_env_by_difficulty(True,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":31}),
    make_config("exp_6_e2_com_r_3v_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 1,**random_env_by_difficulty(False,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":31}),
    make_config("exp_6_f2_no_com_r_3v_easy_thomp", "ppo", {"NUMBER_COMMUNICATION_CHANNELS": 0,**random_env_by_difficulty(False,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":31}),
    ]

exp_4_le_in_choice = [
make_config("exp_6_d_no_com_r_3c_easy_thomp", "ppo", {"FROM_SAVE":True,"NUMBER_COMMUNICATION_CHANNELS": 0,**random_env_by_difficulty(True,3,"very easy"),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
make_config("exp_4_c_com_easy_thomp", "ppo", {"COMMUNISM":True, "XENIA_LOCK": False, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(5,3,False),**thompson_sampling, "EPOCHS": 100000, "SEED":32}),
    make_config("exp_4_o_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 0, **choice_env(4,2, False),**thompson_sampling, "EPOCHS": 1000, "SEED":30}),
    make_config("exp_5_o_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 0, **choice_env(5,3, False),**thompson_sampling, "EPOCHS": 1000, "SEED":30}),
    #make_config("exp_5_a_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(5,3, False),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
    #make_config("exp_5_b_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(5,3, False),**thompson_sampling, "EPOCHS": 10000, "SEED":31}),
    #make_config("exp_5_c_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(5,3, False),**thompson_sampling, "EPOCHS": 10000, "SEED":32}),
    #make_config("exp_5_d_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(5,3, False),**thompson_sampling, "EPOCHS": 10000, "SEED":33}),
    #make_config("exp_4_a_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2),**thompson_sampling, "EPOCHS": 10000, "SEED":30}),
    #make_config("exp_4_b_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2),**thompson_sampling, "EPOCHS": 10000, "SEED":31}),
    #make_config("exp_4_c_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2),**thompson_sampling, "EPOCHS": 10000, "SEED":32}),
    #make_config("exp_4_d_com_easy_thomp", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2),**thompson_sampling, "EPOCHS": 10000, "SEED":33}),
    #make_config("exp_1_a_no_com_easy_eps", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 0, **choice_env(4,2),**epsilon_sampling, "EPOCHS": 1000}),
    #make_config("exp_1_b_com_easy_eps", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(4,2),**epsilon_sampling, "EPOCHS":4000}),
    #make_config("exp_1_c_com_hard_eps", "ppo", {"COMMUNISM":True, "NUMBER_COMMUNICATION_CHANNELS": 1, **choice_env(12,2),**epsilon_sampling}),
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
