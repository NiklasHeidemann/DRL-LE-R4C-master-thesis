from math import ceil

import numpy as np
from plotly.graph_objs import Scatter

from utils.loss_logger import LossLogger, TEST_RETURNS, SOCIAL_REWARD, PREDICTED_GOAL, RETURNS
import plotly.express as px
# code used for the creation of the plots in the thesis. Not intended to be reused, except as inspiration and for reproducibility.

def plot_experiment_4(names: list[str], baselines: list[float|None], dir:str, real_names: list[str], vars=None, y_ranges=None, expname:str="",no_log:bool=False,fileapp:str=""):
    if vars is None:
        vars = [TEST_RETURNS, PREDICTED_GOAL, SOCIAL_REWARD]
        FACTORS = [10,100,1]
    elif vars ==[TEST_RETURNS]:
        FACTORS = [10]
    if y_ranges is None:
        y_ranges = [[0,1],[0,1],[-5,-1]]
    values = {}
    for index, name in enumerate(names):
        logger = LossLogger(run_name=name)
        logger.load(path=f"../experiments/{dir}/logger_")
        values.update({(real_names[index], var):logger.all_smoothed()[var] for var in vars})

    length = len(values[(real_names[0], TEST_RETURNS)]) * 10 if TEST_RETURNS in vars else len(values[(real_names[0], RETURNS)]) * 1
    x_labels = [f"Epoch (tested every 10 epochs)", f"Epoch (tested every 100 epochs)", "Epoch"] if vars!= [RETURNS] else ["Epoch"]
    y_labels = [f"Smoothed test returns", f"Probe classifier accuracy", "Social Influence"]
    if expname:
        if vars == [TEST_RETURNS]:
            titels = [f"{expname}"]
        else:
            titels = [f"Smoothed test returns / {expname}", f"Probe classifier accuracy / {expname}", f"Social Influence / {expname}"]
    else:
        titels = [f"Smoothed test returns", f"Probe classifier accuracy", f"Social Influence"]
    for index, var in enumerate(vars):
        fig = px.line(title=titels[index])
        if y_ranges[index] is not None:
            fig.update_yaxes(range=y_ranges[index])
        if var == SOCIAL_REWARD and not no_log:
            fig.update_yaxes(type="log")
            fig.update_yaxes(tickvals=[10**-x for x in range(6,0,-1)])
        for (name, var2), vals in values.items():
            if var==var2 and len(vals)>0:


                fig.add_trace(Scatter(x=list(range(0,length,FACTORS[index])),y=vals, name=name))
        if baselines[index] is not None:
            fig.add_trace(Scatter(y=[baselines[index]]*length,x=list(range(0,length,1)),name="baseline", line=dict(color="black", dash="dash")))
        fig.update_xaxes(title_text=x_labels[index])
        fig.update_yaxes(title_text=y_labels[index])
        fig.update_layout(font=dict(size=24))

        fig.write_image(f"../experiments/{dir}_{var}{fileapp}.png")


if __name__ == "__main__":
    seed_names = ["seed=30", "seed=31", "seed=32", "seed=33"]
    plot_experiment_4(expname="2 agents",names=["exp_4_a_com_easy_thomp", "exp_4_b_com_easy_thomp", "exp_4_c_com_easy_thomp", "exp_4_d_com_easy_thomp"],baselines=[0.25,0.25,None],dir="LE_chapter5.3",real_names=seed_names)
    plot_experiment_4(expname="oR4C",names=["exp_5_a_com_easy_thomp", "exp_5_b_com_easy_thomp", "exp_5_c_com_easy_thomp", "exp_5_d_com_easy_thomp"],baselines=[1/3,0.2,None],dir="oR4C_chapter5.3",real_names=seed_names)
    plot_experiment_4(expname="env=easy",names=["exp_6_a_com_r_easy_thomp", "exp_6_b_no_com_r_easy_thomp", "exp_6_c_com_r_3c_easy_thomp", "exp_6_d_no_com_r_3c_easy_thomp", "exp_6_e_com_r_3v_easy_thomp","exp_6_f_no_com_r_3v_easy_thomp"],baselines=[None,0.25,None],dir="R4C_easy_chapter5.4",
                      real_names=["Com., 2 ag.","No com., 2 ag.","Com., 3 ag.","No com., 3 ag.","Com., 3 ag.,<br>R4C","No com., 3 ag.,<br>R4C"]),
    plot_experiment_4(expname="env=medium",names=["exp_8_a_com_3c_middle", "exp_8_b_com_3c_middle", "exp_8_c_com_3v_middle","exp_8_d_com_3v_middle" ],y_ranges=[[-1.5,1],[0,1],[-5,-1]],baselines=[None,0.05,None],dir="R4C_medium_chapter5.4",real_names=["coop i)", "coop ii)", "R4C i)", "R4C ii)"]),
    plot_experiment_4(expname="no-com",names=["exp_7_a_no_com_r_middle", "exp_7_b_no_com_r_harder", "exp_7_c_no_com_r_hard"],y_ranges=[[-1.5,1],[0,1],[-5,-1]],baselines=[None,None,None],dir="R4C_nocom_chapter5.4",
                      real_names=["medium", "hard", "very hard"])

    plot_experiment_4(expname="Social Reward Weights",
                      names=["exp_9_a_com_easy_thomp", "exp_9_b_com_easy_thomp", "exp_9_c_com_easy_thomp",
                             "exp_9_d_com_easy_thomp", "exp_9_e_com_easy_thomp", "exp_9_f_com_easy_thomp"],
                      y_ranges=[[-1.5, 1], [0, 1], [-5, -1]], baselines=[0.25, 0.25, None], dir="SocialReward_chapter5.1",
                      real_names=["w=0.01", "w=0.05", "w=0.1", "w=1", "w=10", "w=50"]),
    plot_experiment_4(expname="Test returns / env=choice (com)",names=["sac_choice"],y_ranges=[[-0.2,1]],baselines=[0.25],dir="sac_chapter5.1",real_names=["SAC com."], vars=[TEST_RETURNS],fileapp="_1")
    plot_experiment_4(expname="Test returns / env=random (no. com)",names=["sac_exp_6_b_no_com_r_easy_thomp"],y_ranges=[[-1.5,1.2]],baselines=[2/3],dir="sac_chapter5.1",real_names=["SAC no com."], vars=[TEST_RETURNS],fileapp="_2")
