import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import fsolve, root
from functools import partial
from itertools import product

# Constants
FONTSIZE = 11
COLORS = plt.cm.get_cmap('Set1').colors
colors = [mpl.colors.rgb2hex(color[:3]) for color in COLORS]

# Activation functions
def sigmoid(x, d, a, b):
    return (1 + np.exp(-d * (x - b)))**-1 - (1 + np.exp(d * b))**-1

def leaky_relu(x, d, a, b):
    numerator = (a * x - b)
    denominator = (1 - np.exp(-d * (a * x - b)))
    return np.divide(numerator, denominator, out=np.array(numerator, dtype=float), where=denominator != 0)

def relu(x, d, a, b):
    return np.maximum(x, 0)

def linear(x, d, a, b):
    return (a * x - b)

def identity(x, d, a, b):
    return x

# Plotting utility
def plot_save(axes, k, dyn, params, to_plot):
    range_time = np.arange(params["time_steps"]) * params["dt"]
    ax = axes[0, k]
    if to_plot == "fE":
        ax.plot(range_time, np.mean(dyn["fE1"], axis=0), c=colors[0], ls='--', alpha=0.7, zorder=1)
        ax.plot(range_time, np.mean(dyn["fE2"], axis=0), c=colors[1], ls='-', alpha=0.7, zorder=1)
        ax.fill_between(range_time, np.mean(dyn["fE1"], axis=0) - np.std(dyn["fE1"], axis=0),
                        np.mean(dyn["fE1"], axis=0) + np.std(dyn["fE1"], axis=0), color=colors[0], alpha=0.4, zorder=1)
        ax.fill_between(range_time, np.mean(dyn["fE2"], axis=0) - np.std(dyn["fE2"], axis=0),
                        np.mean(dyn["fE2"], axis=0) + np.std(dyn["fE2"], axis=0), color=colors[1], alpha=0.4, zorder=1)
    elif to_plot == "zE":
        ax.plot(range_time, np.mean(dyn["rE1"], axis=0), c=colors[0], ls='--', alpha=0.7, zorder=1)
        ax.plot(range_time, np.mean(dyn["rE2"], axis=0), c=colors[1], ls='-', alpha=0.7, zorder=1)
        ax.fill_between(range_time, np.mean(dyn["rE1"], axis=0) - np.std(dyn["rE1"], axis=0),
                        np.mean(dyn["rE1"], axis=0) + np.std(dyn["rE1"], axis=0), color=colors[0], alpha=0.4, zorder=1)
        ax.fill_between(range_time, np.mean(dyn["rE2"], axis=0) - np.std(dyn["rE2"], axis=0),
                        np.mean(dyn["rE2"], axis=0) + np.std(dyn["rE2"], axis=0), color=colors[1], alpha=0.4, zorder=1)
    else:
        ax.plot(range_time, np.mean(dyn["rI1"], axis=0), c=colors[0], ls='--', alpha=0.7, zorder=1)
        ax.plot(range_time, np.mean(dyn["rI2"], axis=0), c=colors[1], ls='-', alpha=0.7, zorder=1)
        ax.fill_between(range_time, np.mean(dyn["rI1"], axis=0) - np.std(dyn["rI1"], axis=0),
                        np.mean(dyn["rI1"], axis=0) + np.std(dyn["rI1"], axis=0), color=colors[0], alpha=0.4, zorder=1)
        ax.fill_between(range_time, np.mean(dyn["rI2"], axis=0) - np.std(dyn["rI2"], axis=0),
                        np.mean(dyn["rI2"], axis=0) + np.std(dyn["rI2"], axis=0), color=colors[1], alpha=0.4, zorder=1)
