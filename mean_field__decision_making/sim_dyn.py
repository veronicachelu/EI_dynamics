import numpy as np
from functools import partial

# Helper functions for simulate_neural_dynamics
def compute_stimulus_input(time, stim_start, stim_end, stimulus_strength, coherence):
    """
    Compute the stimulus input for excitatory populations based on time and coherence.
    """
    if stim_start <= time <= stim_end:
        return stimulus_strength * (1 - coherence), stimulus_strength * (1 + coherence)
    return 0, 0

def add_noise(rng, noise_mean, noise_std, tau):
    """
    Add Gaussian noise to the input.
    """
    return rng.normal(noise_mean, noise_std) 

def update_dynamics(current_value, tau, external_input=0, noise=0):
    """
    Update the dynamics of a variable with decay, external input, and noise.
    """
    return -current_value / tau + (external_input + noise) / tau

def compute_validity(params, rE1, rE2, stim_start, stim_end, time_steps, dt):
    """
    Compute the validity of the simulation based on criteria for excitatory populations.
    """
    valid = 0
    invalid_stimulated = True
    for t in range(1, time_steps - 1):
        time = t * dt
        if stim_start > time and (np.abs(rE1[t] - rE2[t]) >= params['valid_tresh'] or
                                  rE1[t] >= params['valid_low_tresh'] or
                                  rE2[t] >= params['valid_low_tresh']):
            valid = 1
            break
        elif stim_start <= time <= stim_end and np.abs(rE1[t] - rE2[t]) >= params['valid_tresh']:
            invalid_stimulated = False
        elif time == stim_end + 1 and np.abs(rE1[t] - rE2[t]) <= params['valid_tresh']:
            valid = 3
    if valid == 0 and invalid_stimulated:
        valid = 2
    return valid

# Main function
def simulate_neural_dynamics(rng, params, f_E, f_I):
    """
    Computes the dynamics of excitatory and inhibitory populations with variable inputs and noise.
    """
    time_steps, dt = params["time_steps"], params["dt"]
    stim_start, stim_end = params["stim_start"], params["stim_end"]
    coherence, stimulus_strength = params["coherence"], params["stimulus_strength"]

    tau_NMDA, tau_AMPA, tau_In, tau_GABA = params["tau_NMDA"], params["tau_AMPA"], params["tau_In"], params["tau_GABA"]
    w_EE, w_EI, w_IE = params["w_values"]["w_EE"], params["w_values"]["w_EI"], params["w_values"]["w_IE"]

    rE1, rE2, rI1, rI2 = np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps)
    rAMPA1, rAMPA2, rIn1, rIn2 = np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps)
    dE1, dE2, dI1, dI2 = np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps)
    In1, In2 = np.zeros(time_steps), np.zeros(time_steps)

    rE1[0], rE2[0], rI1[0], rI2[0] = params["initial_rates"]["E1"], params["initial_rates"]["E2"], params["initial_rates"]["I1"], params["initial_rates"]["I2"]
    rAMPA1[0], rAMPA2[0] = params["initial_rates"]["AMPA"], params["initial_rates"]["AMPA"]

    F_E\
        = partial(f_E, d=params['d'], a=params['a'], b=params['b'])
    F_I = partial(f_I, d=params['d'], a=params['a'], b=params['b'])

    for t in range(1, time_steps - 1):
        time = t * dt
        In1[t], In2[t] = compute_stimulus_input(time, stim_start, stim_end, stimulus_strength, coherence)
        noise1 = add_noise(rng, params["noise_mean"], params["noise_std"], tau_AMPA)
        noise2 = add_noise(rng, params["noise_mean"], params["noise_std"], tau_AMPA)
        dAMPA1 = update_dynamics(rAMPA1[t], tau_AMPA, noise=noise1)
        dAMPA2 = update_dynamics(rAMPA2[t], tau_AMPA, noise=noise2)
        dIn1 = update_dynamics(rIn1[t], tau_In, external_input=In1[t])
        dIn2 = update_dynamics(rIn2[t], tau_In, external_input=In2[t])

        dE1[t] = update_dynamics(rE1[t], tau_NMDA,
                                 external_input=F_E(In1[t] + rAMPA1[t] +
                                                    (1 + w_EE) * rE1[t] + (1 - w_EE) * rE2[t] -
                                                    (1 + w_IE) * rI1[t] - (1 - w_IE) * rI2[t]))
        dE2[t] = update_dynamics(rE2[t], tau_NMDA,
                                 external_input=F_E(In2[t] + rAMPA2[t] +
                                                    (1 + w_EE) * rE2[t] + (1 - w_EE) * rE1[t] -
                                                    (1 + w_IE) * rI2[t] - (1 - w_IE) * rI1[t]))
        dI1[t] = update_dynamics(rI1[t], tau_GABA,
                                 external_input=F_I((1 + w_EI) * rE1[t] + (1 - w_EI) * rE2[t]))
        dI2[t] = update_dynamics(rI2[t], tau_GABA,
                                 external_input=F_I((1 + w_EI) * rE2[t] + (1 - w_EI) * rE1[t]))

        rE1[t + 1], rE2[t + 1] = rE1[t] + dE1[t] * dt, rE2[t] + dE2[t] * dt
        rI1[t + 1], rI2[t + 1] = rI1[t] + dI1[t] * dt, rI2[t] + dI2[t] * dt
        rAMPA1[t + 1], rAMPA2[t + 1] = rAMPA1[t] + dAMPA1 * dt, rAMPA2[t] + dAMPA2 * dt
        rIn1[t + 1], rIn2[t + 1] = rIn1[t] + dIn1 * dt, rIn2[t] + dIn2 * dt

    valid = compute_validity(params, rE1, rE2, stim_start, stim_end, time_steps, dt)

    return {
        "rE1": rE1, "rE2": rE2,
        "rI1": rI1, "rI2": rI2,
        "rAMPA1": rAMPA1, "rAMPA2": rAMPA2,
        "In1": In1, "In2": In2,
        "rIn1": rIn1, "rIn2": rIn2,
    }, valid
