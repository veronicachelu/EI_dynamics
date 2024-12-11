import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Constants
FONTSIZE = 10
COLORS = plt.cm.get_cmap('Set1').colors
colors = [mpl.colors.rgb2hex(color[:3]) for color in COLORS]

# Dictionary to map metrics to labels
dict_of = {
    "avg_reward": "Average Reward",
    "entropy": "Policy Entropy",
    "V_E": r"$V_E$",
    "MSVE": "MSVE",
    "VE_pi": "VE_pi",
    "VE_max": "VE_max",
    "V_I": r"$V_I$",
    "reward": "Reward",
    "action": "Action"
}

def plot_results(params, histories, environment, metric):
    """
    Plot results of the experiments with one subplot.

    Parameters:
        params (dict): Experiment parameters.
        histories (dict): Experiment histories for each condition.
        environment (object): The environment object (for change step info).
        metric (str): The metric to be plotted.
    """
    linestyles = ['-', '--', '-.']
    fig, ax = plt.subplots(figsize=(6, 2), squeeze=True)
    time_steps = params["time_steps"]
    title = f"$w_I$: {params['wI']}, $\\alpha$: {params['alpha']}, $\\tau$: {params['tau']}"
    fig.suptitle(title, fontsize=FONTSIZE)

    for i, (label, history) in enumerate(histories.items()):
        wE = params["wE"][int(label)]
        line_label = f"$w_E$={wE}"
        mean_metric = np.mean(history[metric], axis=0)
        std_metric = np.std(history[metric], axis=0, ddof=1)

        if i == 0:
            ax.plot(np.mean(history["max_reward"], axis=0), label="optimal", color="gray", linestyle=":")
        # Plot the mean with shaded standard deviation
        ax.plot(range(time_steps), mean_metric, label=line_label, color=colors[i], linestyle=linestyles[i], alpha=0.9)
        ax.fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors[i], alpha=0.2)

    # Highlight environment change steps
    for change_step in environment.change_steps:
        ax.axvline(change_step, color="gray", linestyle="--", linewidth=1)

    ax.set_ylabel(dict_of[metric], fontsize=FONTSIZE)
    ax.set_xlabel("Timesteps", fontsize=FONTSIZE)
    ax.legend(loc='lower left', fontsize=FONTSIZE, handlelength=1, handletextpad=0.1)
    plt.show()

def plot_results_2cols(params, histories, environment, metric):
    """
    Plot results of the experiments with two subplots.

    Parameters:
        params (dict): Experiment parameters.
        histories (dict): Experiment histories for each condition.
        environment (object): The environment object (for change step info).
        metric (str): The metric to be plotted.
    """
    linestyles = ['-', '--', '-.']
    fig, axes = plt.subplots(1, 2, figsize=(12, 2), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    time_steps = params["time_steps"]
    title = f"$w_E$: {params['wE']}, $\\alpha$: {params['alpha']}, $\\tau$: {params['tau']}"
    fig.suptitle(title, fontsize=FONTSIZE)

    for i, (label, history) in enumerate(histories.items()):
        wI = params["wI"][int(label)]
        line_label = f"$w_I$={wI}"
        mean_metric = np.mean(history[metric], axis=0)
        std_metric = np.std(history[metric], axis=0, ddof=1)

        if i == 0 and metric == "avg_reward":
            axes[0].plot(np.mean(history["max_reward"], axis=0), label="optimal", color="gray", linestyle=":")
            axes[1].plot(np.mean(history["max_reward"], axis=0), label="optimal", color="gray", linestyle=":")

        if i == 1 or i == 0:
            # Plot for first subplot
            axes[0].plot(range(time_steps), mean_metric, label=line_label, color=colors[i], linestyle=linestyles[i], alpha=0.9)
            axes[0].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors[i], alpha=0.2)
        if i == 2 or i == 0:
            # Plot for second subplot
            axes[1].plot(range(time_steps), mean_metric, label=line_label, color=colors[i], linestyle=linestyles[i], alpha=0.9)
            axes[1].fill_between(range(time_steps), mean_metric - std_metric, mean_metric + std_metric, color=colors[i], alpha=0.2)

    # Highlight environment change steps
    for change_step in environment.change_steps:
        for ax in axes:
            ax.axvline(change_step, color="gray", linestyle="--", linewidth=1)

    axes[0].set_ylabel(dict_of[metric], fontsize=FONTSIZE)
    axes[0].set_xlabel("Timesteps", fontsize=FONTSIZE)
    axes[1].set_xlabel("Timesteps", fontsize=FONTSIZE)
    axes[0].legend(loc='lower left', fontsize=FONTSIZE, handlelength=1, handletextpad=0.1)
    axes[1].legend(loc='lower left', fontsize=FONTSIZE, handlelength=1, handletextpad=0.1)
    plt.show()
