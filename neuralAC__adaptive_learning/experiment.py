import numpy as np
from collections import defaultdict
from utils import compute_entropy
from environment import generate_task_distribution, ContinualBandit
from neural_actor_critic import initialize_policy, credit_assignment, mirror_descent_with_entropy


def run_experiment(rng, n_actions, num_steps, environment, wE, alpha, wI, tau, verbose=0):
    """
    Runs a single experiment and logs various metrics at each step.

    Parameters:
        rng (np.random.Generator): Random number generator for reproducibility.
        n_actions (int): Number of available actions.
        num_steps (int): Total number of steps in the experiment.
        environment (ContinualBandit): The bandit environment.
        wE (float): Weight of exploratory value.
        alpha (float): Learning rate.
        wI (float): Weight of exploitative influence.
        tau (float): Entropy regularization coefficient.
        verbose (int): Verbosity level for logging progress.

    Returns:
        defaultdict: Experiment history containing metrics at each step.
    """
    policy = initialize_policy(n_actions)
    Q_E = np.zeros(n_actions)
    Q_I = np.zeros(n_actions)
    action_counts = np.zeros(n_actions)

    history = defaultdict(list)

    for step in range(num_steps):
        # Select an action based on the current policy
        action = rng.choice(n_actions, p=policy)

        # Step through the environment and receive a reward
        reward = environment.step(action, step)

        # Update the reward estimates
        action_counts[action] += 1
        Q_E, Q_I = credit_assignment(reward, action, Q_E, Q_I, alpha, wI)

        # Update the policy using mirror descent with entropy regularization
        policy = mirror_descent_with_entropy(policy, Q_E, wE, tau)

        # Log metrics
        exp_r = environment.expected_rewards()
        history["reward"].append(reward)
        history["MSVE"].append(np.mean((exp_r - Q_E) ** 2))
        history["VE_pi"].append(np.sum(policy * (exp_r - Q_E) ** 2))
        history["VE_max"].append(np.sum(np.max(exp_r) * (exp_r - Q_E) ** 2))
        history["action"].append(action)
        history["Q_E"].append(Q_E[action])
        history["V_E"].append(np.dot(policy, Q_E))
        history["Q_I"].append(Q_I[action])
        history["V_I"].append(np.dot(policy, Q_I))
        history["max_reward"].append(np.max(exp_r))
        history["avg_reward"].append(np.dot(policy, exp_r))
        history["entropy"].append(compute_entropy(policy))

        if verbose and step % (num_steps // 10) == 0:
            print(f"Step {step}/{num_steps} - Avg Reward: {history['avg_reward'][-1]:.4f}")

    return history

def sensitivity_sweep(params, sweep_over="wE"):
    """
    Performs a sensitivity sweep over the specified parameter (wE or wI).

    Parameters:
        params (dict): Experiment parameters.
        sweep_over (str): The parameter to sweep over ("wE" or "wI").

    Returns:
        tuple: Histories of experiments and the environment instance.
    """
    if sweep_over not in ["wE", "wI"]:
        raise ValueError(f"Invalid sweep parameter: {sweep_over}. Must be 'wE' or 'wI'.")

    n_actions = 2
    intervals = [params["time_steps"] // params["n_tasks"]] * params["n_tasks"]
    tasks = generate_task_distribution(
        task_dist=params["task_dist"],
        n_tasks=params["n_tasks"],
        init_reward_means=params["init_reward_means"],
        init_reward_probs=params["init_reward_probs"],
        task_reward_probs=params["task_reward_probs"]
    )

    histories = {}
    alpha, tau = params["alpha"], params["tau"]
    sweep_values = params[sweep_over]

    for i, sweep_value in enumerate(sweep_values):
        histories[i] = defaultdict(list)

        for seed in range(params["n_trials"]):
            rng = np.random.default_rng(seed)
            env = ContinualBandit(
                rng=rng,
                n_actions=n_actions,
                intervals=intervals,
                tasks=tasks,
                reward_prob=params["reward_prob"],
                reward_var=params["reward_var"]
            )

            # Choose the appropriate parameter to sweep
            if sweep_over == "wE":
                wE, wI = sweep_value, params["wI"]
            else:  # sweep_over == "wI"
                wE, wI = params["wE"], sweep_value

            history = run_experiment(rng, n_actions, params["time_steps"], env, wE, alpha, wI, tau)
            for key, value in history.items():
                histories[i][key].append(value)

    return histories, env

