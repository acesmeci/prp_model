import numpy as np
# Dont' change lambda = 0.4, alpha = 0.2, beta = 0.2, noise_std = 0.2, t0 = 0.15
# These are parameter used in the paper
def run_lca(input_series,
            relevant_output_indices,
            dt=0.1,
            max_timesteps=100,
            lambda_=0.4, # 0.4 
            alpha=0.2, # 0.2
            beta=0.2, # 0.2
            noise_std=0.2, # 0.2
            threshold=1.0, # Figure out threshold based on the paper
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
                noise_std=0.2, threshold=1.0, t0=0.15):
    
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

