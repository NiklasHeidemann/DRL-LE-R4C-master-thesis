from plotly.graph_objs import Scatter

from utils.loss_logger import LossLogger, TEST_RETURNS, SOCIAL_REWARD, PREDICTED_GOAL
import plotly.express as px

def plot_experiment_4(names: list[str],baselines: list[float|None],dir:str,real_names: list[str]):
    vars = [TEST_RETURNS, PREDICTED_GOAL, SOCIAL_REWARD]
    values = {}
    for index, name in enumerate(names):
        logger = LossLogger(run_name=name)
        logger.load(path=f"../experiments/{dir}/logger_")
        values.update({(real_names[index], var):logger.all_smoothed()[var] for var in vars})

    x_labels = [f"epoch (tested every 10 epochs)", f"epoch (tested every 100 epochs)", "epoch"]
    y_labels = [f"Smoothed test returns", f"Predicted goal", "Social Influence"]
    titels = [f"Smoothed test returns after 10.000 epochs", f"Predicted goal after 10.000 epochs", "Social Influence after 10.000 epochs"]

    for index, var in enumerate(vars):
        fig = px.line(title=titels[index])
        for (name, var2), vals in values.items():
            if var==var2 and len(vals)>0:
                fig.add_trace(Scatter(x=list(range(0,10000,10000//len(vals))),y=vals, name=name))
        if baselines[index] is not None:
            fig.add_trace(Scatter(y=[baselines[index]]*10000,x=list(range(0,10000,1)),name="baseline", line=dict(color="black", dash="dash")))
        fig.update_xaxes(title_text=x_labels[index])
        fig.update_yaxes(title_text=y_labels[index])

        fig.write_image(f"../experiments/{dir}_{var}.png")


if __name__ == "__main__":
    seed_names = ["seed=30", "seed=31", "seed=32", "seed=33"]
    #plot_experiment_4(names=["exp_4_a_com_easy_thomp", "exp_4_b_com_easy_thomp", "exp_4_c_com_easy_thomp", "exp_4_d_com_easy_thomp"],baselines=[0.25,0.25,None],dir="exp4",real_names=seed_names)
    #plot_experiment_4(names=["exp_5_a_com_easy_thomp", "exp_5_b_com_easy_thomp", "exp_5_c_com_easy_thomp", "exp_5_d_com_easy_thomp"],baselines=[1/3,0.2,None],dir="exp5",real_names=seed_names)
    plot_experiment_4(names=["exp_6_a_com_r_easy_thomp", "exp_6_b_no_com_r_easy_thomp", "exp_6_c_com_r_3c_easy_thomp", "exp_6_d_no_com_r_3c_easy_thomp", "exp_6_e_com_r_3v_easy_thomp","exp_6_f_no_com_r_3v_easy_thomp"],baselines=[None,0.25,None],dir="exp6",
                      real_names=["Com., 2 agents", "No com., 2 agents", "Com., 3 agents, coop", "No com., 3 agents, coop", "Com., 3 agents, versus", "No com., 3 agents, versus"])