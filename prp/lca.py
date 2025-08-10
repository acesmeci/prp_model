import numpy as np
# Don't change lambda = 0.4, alpha = 0.2, beta = 0.2, noise_std = 0.2, t0 = 0.15
# These are the parameters used in the paper
def run_lca(input_series,
            relevant_output_indices,
            dt=0.05, # 0.1 in MATLAB
            max_timesteps=100,
            lambda_=0.4, # 0.4 
            alpha=0.2, # 0.2
            beta=0.2, # 0.2
            noise_std=0.1, # 0.2 in paper, 0.1 in MATLAB
            threshold=1.0, # Decided by reward maximization (optimize_lca_threshold)
            t0=0.15 # 0.15
           ):
    """
    Simulates LCA dynamics over time for a set of output units.

    Args:
        input_series: list of activation vectors (length = time steps)
        relevant_output_indices: list of indices (e.g., [3,4,5]) for the relevant output dimension
        dt: time step duration
        max_timesteps: max simulation length
        lambda_: leak
        alpha: self-excitation
        beta: lateral inhibition
        noise_std: noise strength
        threshold: decision threshold
        t0: non-decision time

    Returns:
        rt: decision time in continuous units (or None)
        choice: which unit crossed threshold (or None)
        trajectory: activation over time for relevant units
    """
    n_units = len(relevant_output_indices)
    x = np.zeros(n_units)  # accumulator activations
    trajectory = []
    input_series = np.array(input_series)  # convert to numpy array if not already
    
    for t in range(min(len(input_series), max_timesteps)):
        inp = input_series[t][relevant_output_indices]
        noise = np.random.randn(n_units) * noise_std

        dx = -lambda_ * x + alpha * x + inp - beta * (np.sum(x) - x) + noise
        x += dt * dx
        trajectory.append(x.copy())

        if np.any(x >= threshold):
            rt = t * dt + t0
            choice = np.argmax(x)
            return rt, choice, trajectory

    return None, None, trajectory

# Get RTs and accuracy across 100 simulations, instead of just one
def run_lca_avg(input_series, relevant_output_indices,
                n_repeats=100, dt=0.1, max_timesteps=100,
                lambda_=0.4, alpha=0.2, beta=0.2,
                noise_std=0.1, # 0.2 in paper, 0.1 in MATLAB
                threshold=1.0, t0=0.15):
    
    rts = []
    corrects = []
    for _ in range(n_repeats):
        rt, choice, _ = run_lca(
            input_series,
            relevant_output_indices,
            dt=dt,
            max_timesteps=max_timesteps,
            lambda_=lambda_,
            alpha=alpha,
            beta=beta,
            noise_std=noise_std,
            threshold=threshold,
            t0=t0
        )
        if rt is not None:
            rts.append(rt)
            corrects.append(choice)

    if not rts:
        return None, None  # No threshold crossings

    avg_rt = np.mean(rts)
    most_common_choice = max(set(corrects), key=corrects.count)
    return avg_rt, most_common_choice


# MATLAB implementation of runLCA with RR optimization (NNmodel.m line 1124-1715)
def run_lca_dist(
    input_series,
    relevant_output_indices,
    thresholds=np.arange(0.0, 1.6, 0.1),
    n_repeats=100,
    dt=0.01,
    tau=0.1,
    lambda_=0.4,
    alpha=0.2,
    beta=0.2,
    noise_std=0.1,
    t0=0.15
):
    """
    Simulates full LCA dynamics for all thresholds.

    Args:
        input_series: shape [T, output_dim] (output activations)
        relevant_output_indices: indices of the output units for this task
        thresholds: list/array of threshold values
        n_repeats: number of LCA simulations per threshold

    Returns:
        result_dict with keys:
            'thresholds', 'reward_rates', 'accuracies', 'rts',
            'rts_correct', 'rts_incorrect', 'all_choices'
    """
    input_series = np.array(input_series)
    output_dim = input_series.shape[1]
    n_steps = input_series.shape[0]
    n_thresholds = len(thresholds)
    n_units = len(relevant_output_indices)

    input_series = input_series[:, relevant_output_indices]  # [T, units]

    # Precompute external input (constant across sims)
    p = input_series  # shape [T, units]

    # Lateral inhibition matrix
    W_inhib = -np.ones((n_units, n_units)) + np.eye(n_units)

    # Storage
    all_rts = np.full((n_thresholds, n_repeats), np.nan)
    all_accs = np.zeros((n_thresholds, n_repeats))
    all_choices = np.full((n_thresholds, n_repeats), -1)

    dt_tau = dt / tau
    sqrt_dt_tau = np.sqrt(dt_tau)

    for thresh_idx, z in enumerate(thresholds):
        for sim in range(n_repeats):
            x = np.zeros(n_units)  # internal state
            f = np.zeros(n_units)  # activation
            noise = noise_std * np.random.randn(n_steps, n_units)

            rt = None
            choice = None

            for t in range(n_steps):
                I_ext = p[t]
                lateral = beta * f @ W_inhib
                dx = (I_ext - lambda_ * x + alpha * f + lateral) * dt_tau + noise[t] * sqrt_dt_tau
                x += dx
                f = np.maximum(x, 0)

                above_thresh = np.where(f > z)[0]
                if len(above_thresh) > 0:
                    choice = np.random.choice(above_thresh)
                    rt = t * dt + t0
                    break

            if rt is not None:
                all_rts[thresh_idx, sim] = rt
                all_choices[thresh_idx, sim] = choice
                correct = (choice == np.argmax(p[0]))  # first time point = true label
                all_accs[thresh_idx, sim] = int(correct)

    # Compute stats
    accs = np.nanmean(all_accs, axis=1)
    rts = np.nanmean(all_rts, axis=1)
    reward_rates = accs / (rts + 1e-5)  # prevent div by 0

    return {
        'thresholds': thresholds,
        'reward_rates': reward_rates,
        'accuracies': accs,
        'rts': rts,
        'all_choices': all_choices,
        'all_rts': all_rts,
        'all_accs': all_accs
    }

