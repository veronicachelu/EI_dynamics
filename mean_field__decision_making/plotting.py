import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import plot_save
from sim_dyn import simulate_neural_dynamics
from functools import partial
from itertools import product
import copy

FONTSIZE = 11
def initialize_logs(keys):
    """
    Initialize a dictionary of logs for storing simulation results.
    """
    return {k: [] for k in keys}


def simulate_single_trial(params, stim_start, stim_end, w_EE, f_E, f_I, rng):
    """
    Run a single trial of the simulation.
    """
    params["stim_start"] = stim_start
    params["stim_end"] = stim_end
    params["w_values"]["w_EE"] = w_EE

    return simulate_neural_dynamics(rng, params, f_E, f_I)


def setup_axes(fig, axes, params, m):
    """
    Set up the axes and labels for the simulation plot.
    """
    for j in range(m):
        stim_start = params["stimulation_period"][0] * params["time_steps"] * params["dt"]
        stim_end = params["stimulation_period"][1] * params["time_steps"] * params["dt"]
        axes[0, j].axvspan(stim_start, stim_end, color="gray", alpha=0.3)
    return fig, axes


def plot_trial_results(axes, j, logs, params, to_plot):
    """
    Plot the results of a single trial.
    """
    plot_save(axes, j, logs, params, to_plot)


def plot_simulation(params, f_E, f_I, to_plot="fE", use_full_combinations=False):
    """
    Simulate neural dynamics and plot the results for multiple connectivity configurations.
    Supports two modes:
    - `use_full_combinations=False`: Simulates for coherence levels and excitatory-to-excitatory connectivity.
    - `use_full_combinations=True`: Simulates for all combinations of connectivity parameters.
    """
    from itertools import product

    keys = ["fE1", "fE2", "rE1", "rE2", "dE1", "dE2", "dI1", "dI2",
            "In1", "In2", "rI1", "rI2", "rIn1", "rIn2", "rAMPA1", "rAMPA2", "dAMPA1", "dAMPA2"]

    start_t = int(params["stimulation_period"][0] * params["time_steps"])
    end_t = int(params["stimulation_period"][1] * params["time_steps"])
    stim_start, stim_end = start_t * params["dt"], end_t * params["dt"]

    if use_full_combinations:
        # Compute full combinations of w_EE, w_EI, w_IE
        combinations = list(product(params["w_values"]["w_EE"], params["w_values"]["w_EI"], params["w_values"]["w_IE"]))
        m = len(combinations)
    else:
        # Only use coherence levels and excitatory-to-excitatory connectivity
        combinations = [(w_EE, None, None) for w_EE in params["w_values"]["w_EE"]]
        m = len(params["coherence_list"])

    fig, axes = plt.subplots(1, m, figsize=(4 * m, 2), sharex='col', sharey='row', squeeze=False, gridspec_kw={'hspace': 0.1})

    valid_trials = []
    for j, combo in enumerate(combinations):
        logs = initialize_logs(keys)
        valid_count = 0
        trial_types = []
        seed = -1

        # Extract connectivity parameters
        w_EE = combo[0]
        w_EI = combo[1] if use_full_combinations else params["w_values"]["w_EI"][0]
        w_IE = combo[2] if use_full_combinations else params["w_values"]["w_IE"][0]

        for trial in range(params["n_trials"]):
            seed += 1
            rng = np.random.default_rng(seed)
            params_single = copy.deepcopy(params)
            params_single["stim_start"] = stim_start
            params_single["stim_end"] = stim_end
            params_single["w_values"]["w_EE"] = w_EE
            params_single["w_values"]["w_EI"] = w_EI
            params_single["w_values"]["w_IE"] = w_IE

            if not use_full_combinations:
                params_single["coherence"] = params["coherence_list"][j % len(params["coherence_list"])]
            else:
                params_single["coherence"] = rng.uniform(params["coherence_range"][0], params["coherence_range"][1])

            # Simulate dynamics
            dyn, valid_type = simulate_neural_dynamics(rng, params_single, f_E, f_I)
            trial_types.append(valid_type)

            if valid_type == 0:
                valid_count += 1

            for k, v in dyn.items():
                logs[k].append(v)

        valid_trials.append(valid_count / params["n_trials"])

        # Plot results
        axes[0, j].axvspan(stim_start, stim_end, color="gray", alpha=0.3)
        title = f"$w_{{EE}}$: {w_EE}"
        if use_full_combinations:
            title += f", $w_{{EI}}$: {w_EI}, $w_{{IE}}$: {w_IE}"
        title += f"\n(valid: {valid_trials[-1]:.2f})"
        axes[0, j].set_title(title, fontsize=11)
        axes[0, j].set_xlabel("Time (s)", fontsize=11)

        plot_save(axes, j, logs, params, to_plot)

    # Add y-axis labels
    ylabel_map = {
        "fE": r"$f(\mathbf{z}_{\mathrm{E}})$",
        "zE": r"$\mathbf{z}_{\mathrm{E}}$",
        "zI": r"$\mathbf{z}_{\mathrm{I}}$"
    }
    axes[0, 0].set_ylabel(ylabel_map.get(to_plot, ""), fontsize=11)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    return valid_trials
