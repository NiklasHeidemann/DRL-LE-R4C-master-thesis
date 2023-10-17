import math
import os
from typing import Sequence, Mapping

import plotly.express as px
from plotly.graph_objs import Scatter
from plotly.subplots import make_subplots


def plot_returns(run_name: str, returns: Sequence[float], identifier: str ="returns")->Scatter:
    returns = returns[:] # due to parallel processing
    x = list(range(len(returns)))
    """plt.plot(x,returns)
    plt.title(f"Smoothed {identifier} after {len(returns)} episodes")
    plt.savefig(f"plots/{run_name}_{identifier}.png")
    plt.close()"""
    fig = px.line(x=x, y=returns, title=f"Smoothed {identifier} after {len(returns)} episodes", labels={"x":"episode", "y":f"Smoothed {identifier}"})
    fig.write_image(f"plots/{run_name}_{identifier}.png")
    return Scatter(x=x, y=returns)
def plot_multiple(run_name: str, values: Mapping[str, Sequence[float]])->None:
    size = math.ceil(math.sqrt(len(values)))
    main_fig = make_subplots(rows=size, cols=size, subplot_titles=list(values.keys()))
    counter = 0
    for identifier, returns in values.items():
        if len(returns) == 0:
            continue
        fig = plot_returns(run_name=run_name, returns=returns, identifier=identifier)
        main_fig.add_trace(fig, row=counter//size+1, col=counter%size+1)
        main_fig.update_xaxes(title_text="episode", row=counter//size+1, col=counter%size+1)
        main_fig.update_yaxes(title_text=f"Smoothed {identifier}", row=counter//size+1, col=counter%size+1)
        counter += 1
    main_fig.update_layout(title_text=f"All smoothed values after {max([len(list_) for list_ in values.values()])} episodes", height=300*size, width=400*size, showlegend=False)
    main_fig.write_image(f"plots/__{run_name}_all.png")
    print("plotting thread end")
def delete_old_plots():
    for path in os.listdir("."):
        if path.endswith(".png") and not path.startswith("_"):
            os.remove(path)