# Legacy threshold optimizers (currently unused, kept for fallback/debug):
# - optimize_reward_rate_threshold
# - optimize_lca_threshold

import numpy as np
from prp.lca import run_lca_avg
from prp.lca import run_lca_dist

DEFAULT_N_REPEATS = 100 # Default number of repeats for LCA simulations. Use 100 if you have access to a GPU.

# Not used in the current implementation. Replaced by optimize_lca_threshold_dist.
def optimize_lca_threshold(input_series, relevant_output_indices, correct_response_idx,
                           thresholds=np.arange(0.0, 1.6, 0.1),
                           ITI=4.0,
                           n_repeats=DEFAULT_N_REPEATS):
    """
    Finds LCA threshold z that maximizes reward rate: acc / (ITI + RT)
    Based on full distribution over n_repeats using run_lca_dist().
    """
    results = run_lca_dist(
        input_series=input_series,
        relevant_output_indices=relevant_output_indices,
        thresholds=thresholds,
        n_repeats=n_repeats
    )

    reward_rates = results['reward_rates']
    best_idx = np.argmax(reward_rates)
    best_threshold = results['thresholds'][best_idx]

    return best_threshold


# Optimie LCA threshold with run_lca_dist. Faithful to MATLAB implementation
# Correct_response_idx is not used directly here, because run_lca_dist() infers correctness from argmax(p[0]). Change this if needed.
def optimize_lca_threshold_dist(
    input_series,
    relevant_output_indices,
    correct_response_idx=None,
    thresholds=np.arange(0.0, 1.6, 0.1),
    ITI=0.5, # Paper: 0.5, MATLAB: 4.0
    n_repeats=DEFAULT_N_REPEATS,
    dt=0.01,
    tau=0.1,
    lambda_=0.4,
    alpha=0.2,
    beta=0.2,
    noise_std=0.1,
    t0=0.15,
    verbose=False
):
    """
    Sweeps across thresholds to maximize reward rate using full LCA distribution.

    Args:
        input_series: time series of model output activations
        relevant_output_indices: output units corresponding to the current task
        correct_response_idx: correct unit index (relative to relevant_output_indices); if None, inferred via argmax
        thresholds: range of thresholds to test
        ITI: inter-trial interval
        n_repeats: number of LCA runs per threshold
        dt, tau, lambda_, alpha, beta, noise_std, t0: LCA parameters
        verbose: whether to print full sweep

    Returns:
        best_threshold: threshold that maximizes reward rate
        summary: full result dictionary from run_lca_dist
    """
    results = run_lca_dist(
        input_series=input_series,
        relevant_output_indices=relevant_output_indices,
        thresholds=thresholds,
        n_repeats=DEFAULT_N_REPEATS,
        dt=dt,
        tau=tau,
        lambda_=lambda_,
        alpha=alpha,
        beta=beta,
        noise_std=noise_std,
        t0=t0
    )

    reward_rates = results['reward_rates']
    accs = results['accuracies']
    rts = results['rts']
    thresh_list = results['thresholds']

    best_idx = np.argmax(reward_rates)
    best_threshold = thresh_list[best_idx]

    if verbose:
        for i in range(len(thresh_list)):
            print(f"z={thresh_list[i]:.2f} | Acc={accs[i]:.2f} | RT={rts[i]:.3f} | RR={reward_rates[i]:.3f}")
        print(f"✅ Best threshold z: {best_threshold:.2f}")

    return best_threshold, results

# Updated version of choose_onset_policy to use optimize_lca_threshold_dist
# Can add verbose if needed
def choose_onset_policy(task_net, input_a, input_b, task_a, task_b,
                        max_onset_delay=15, soa=3,
                        n_repeats=DEFAULT_N_REPEATS, # it was 20 before
                        tau_net=0.2, tau_task=0.2, persistence=0.5,
                        ITI=0.5):
    """
    Searches for Task 2 onset that maximizes joint reward rate using dynamic threshold optimization.

    Returns:
        optimal_onset (int): time step to start Task 2
    """
    best_rr = -np.inf
    best_onset = soa  # default to SOA if nothing better found
    N_pathways = 3
    N_features = 3

    # Decode task structure
    task_matrix_b = task_b.reshape(N_pathways, N_pathways).T
    in_b, out_b = np.argwhere(task_matrix_b == 1)[0]
    output_b = list(range(out_b * N_features, (out_b + 1) * N_features))
    correct_b = np.argmax(input_b[in_b * N_features : (in_b + 1) * N_features])

    task_matrix_a = task_a.reshape(N_pathways, N_pathways).T
    in_a, out_a = np.argwhere(task_matrix_a == 1)[0]
    output_a = list(range(out_a * N_features, (out_a + 1) * N_features))
    correct_a = np.argmax(input_a[in_a * N_features : (in_a + 1) * N_features])

    for delay in range(soa, max_onset_delay + 1):
        input_series = []
        task_series = []

        for t in range(delay + 50):  # simulate long enough to allow both decisions
            stim_t = np.zeros_like(input_a)
            task_t = np.zeros_like(task_a)

            if t >= 0:
                stim_t += input_a
                task_t += task_a
            if t >= delay:
                stim_t += input_b
                task_t += task_b

            input_series.append(stim_t)
            task_series.append(task_t)

        output_series = task_net.integrate(
            input_series, task_series,
            tau_net=tau_net, tau_task=tau_task, persistence=persistence
        )

        # === Optimize thresholds per task ===
        z_a, _ = optimize_lca_threshold_dist(output_series, output_a, correct_response_idx=correct_a, ITI=ITI, n_repeats=n_repeats)
        z_b, _ = optimize_lca_threshold_dist(output_series[delay:], output_b, correct_response_idx=correct_b, ITI=ITI, n_repeats=n_repeats)

        # === Evaluate performance ===
        rt_a, choice_a = run_lca_avg(output_series, output_a, threshold=z_a, n_repeats=n_repeats)
        acc_a = int(choice_a == correct_a) if choice_a is not None else 0

        rt_b, choice_b = run_lca_avg(output_series[delay:], output_b, threshold=z_b, n_repeats=n_repeats)
        acc_b = int(choice_b == correct_b) if choice_b is not None else 0

        if rt_a is None or rt_b is None:
            continue

        # PRP-style joint reward rate
        rt_total = max(rt_a, delay * 0.1 + rt_b)
        reward_rate = (acc_a * acc_b) / (ITI + rt_total)

        if reward_rate > best_rr:
            best_rr = reward_rate
            best_onset = delay

    return best_onset


# Finds best Task A threshold *z* ahead of time to be used in sweep_soa()
def optimize_reward_rate_threshold(net, input_a, input_b, task_a, task_b,
                                   soa, max_timesteps=100,
                                   thresholds=np.arange(0.0, 1.6, 0.1),
                                   tau_net=0.2, tau_task=0.2, persistence=0.5,
                                   ITI=0.5,
                                   n_repeats=DEFAULT_N_REPEATS):
    """
    Runs a simulated PRP trial and finds the threshold z that maximizes reward rate
    for Task A (the first task). Used in sweep_soa(), BEFORE running full trials.
    """
    input_dim = input_a.shape[0]
    task_dim = task_a.shape[0]

    onset_b = choose_onset_policy(
        task_net=net,
        input_a=input_a,
        input_b=input_b,
        task_a=task_a,
        task_b=task_b,
        soa=soa,
        tau_net=tau_net,
        tau_task=tau_task,
        persistence=persistence,
        ITI=ITI
    )

    # Create full input + task series
    input_series, task_series = [], []
    for t in range(max_timesteps):
        stim_t = np.zeros(input_dim)
        task_t = np.zeros(task_dim)

        if t >= 0:
            stim_t += input_a
            task_t += task_a
        if t >= onset_b:
            stim_t += input_b
            task_t += task_b

        input_series.append(stim_t)
        task_series.append(task_t)

    # Run integration
    output_series = net.integrate(
        input_series, task_series,
        tau_net=tau_net,
        tau_task=tau_task,
        persistence=persistence
    )

    # Target output for Task A (first)
    N_pathways = 3
    N_features = 3
    task_matrix = task_a.reshape(N_pathways, N_pathways).T
    in_dim, out_dim = np.argwhere(task_matrix == 1)[0]
    output_idxs = list(range(out_dim * N_features, (out_dim + 1) * N_features))
    correct_idx = np.argmax(input_a[in_dim * N_features:(in_dim + 1) * N_features])

    # Use full LCA reward-rate scan
    best_z, _ = optimize_lca_threshold_dist(
        output_series=output_series,
        relevant_output_indices=output_idxs,
        correct_response_idx=correct_idx,
        thresholds=thresholds,
        ITI=ITI,
        n_repeats=n_repeats,
    )

    return best_z
