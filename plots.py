import os
from typing import Sequence, Mapping

import matplotlib.pyplot as plt

def plot_returns(returns: Sequence[float], identifier: str ="returns")->None:
    returns = returns[:] # due to parallel processing
    x = list(range(len(returns)))
    plt.plot(x,returns)
    plt.title(f"Smoothed {identifier} after {len(returns)} episodes")
    plt.savefig(f"plots/{identifier}.png")
    plt.close()
    print("plot returns")
def plot_multiple(values: Mapping[str, Sequence[float]])->None:
    for identifier, returns in values.items():
        plot_returns(returns=returns, identifier=identifier)
    print("threads end")
def delete_old_plots():
    for path in os.listdir("."):
        if path.endswith(".png") and not path.startswith("_"):
            os.remove(path)