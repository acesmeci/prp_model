import numpy as np
# Don't change lambda = 0.4, alpha = 0.2, beta = 0.2, noise_std = 0.2, t0 = 0.15
# These are the parameters used in the paper
def run_lca(input_series,
            relevant_output_indices,
            dt=0.1, # 0.1 in MATLAB
            max_timesteps=100, # 1000 in MATLAB?
            lambda_=0.4, # 0.4 
            alpha=0.2, # 0.2
            beta=0.2, # 0.2
            noise_std=0.2, # 0.2 in paper, 0.1 in MATLAB
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
# This function doesn't use ITI in the reward rate calculation. Was this the case in the MATLAB code?
def run_lca_dist(
    input_series,
    relevant_output_indices,
    thresholds=np.arange(0.1, 1.6, 0.1),
    n_repeats=100,
    dt=0.1,
    tau=0.1,
    lambda_=0.4,
    alpha=0.2,
    beta=0.2,
    noise_std=0.2,
    t0=0.15,
    ITI=0.5,
    correct_response_idx=None,   # <-- NEW: true label (relative to relevant indices)
):
    """
    Simulates full LCA dynamics for all thresholds and returns summary stats.
    - If the accumulator never hits the threshold, RT is NaN and accuracy=0.
    - Reward-rate is set to 0 when RT is NaN (so those z can't "win").
    """
    import numpy as np

    input_series = np.array(input_series)
    p = input_series[:, relevant_output_indices]          # [T, n_units]
    n_steps, n_units = p.shape
    n_thresholds = len(thresholds)

    # Lateral inhibition matrix
    W_inhib = -np.ones((n_units, n_units)) + np.eye(n_units)

    # Storage
    all_rts = np.full((n_thresholds, n_repeats), np.nan, dtype=float)
    all_accs = np.zeros((n_thresholds, n_repeats), dtype=float)

    dt_tau = dt / tau
    sqrt_dt_tau = np.sqrt(dt_tau)

    # Label to score against
    if correct_response_idx is None:
        # fallback: most-active at t=0 within relevant block
        correct_response_idx = int(np.argmax(p[0]))

    for ti, z in enumerate(thresholds):
        for rep in range(n_repeats):
            x = np.zeros(n_units)  # state
            f = np.zeros(n_units)  # rectified activation
            noise = noise_std * np.random.randn(n_steps, n_units)

            rt = np.nan
            choice = -1

            for t in range(n_steps):
                I_ext = p[t]
                lateral = beta * f @ W_inhib
                dx = (I_ext - lambda_ * x + alpha * f + lateral) * dt_tau + noise[t] * sqrt_dt_tau
                x += dx
                f = np.maximum(x, 0.0)

                above = np.where(f > z)[0]
                if len(above) > 0:
                    choice = int(np.random.choice(above))
                    rt = t * dt + t0
                    break

            all_rts[ti, rep] = rt
            if not np.isnan(rt):
                all_accs[ti, rep] = 1.0 if (choice == correct_response_idx) else 0.0
            # else accuracy stays 0

    # Aggregate
    accs = np.nanmean(all_accs, axis=1)                   # [n_thresholds]
    rts  = np.nanmean(all_rts,  axis=1)                   # [n_thresholds]

    # Reward-rate: 0 when no decision (rts is NaN)
    reward_rates = np.zeros_like(rts, dtype=float)
    valid = ~np.isnan(rts)
    reward_rates[valid] = accs[valid] / (ITI + rts[valid])

    return {
        "thresholds": np.array(thresholds, dtype=float),
        "reward_rates": reward_rates,
        "accuracies": accs,
        "rts": rts,
        "all_choices": None,
        "all_rts": all_rts,
        "all_accs": all_accs,
    }
