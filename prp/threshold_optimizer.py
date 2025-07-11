import numpy as np
from prp.lca import run_lca_avg
from prp.choose_onset_policy import choose_onset_policy


def optimize_lca_threshold(input_series, relevant_output_indices, correct_response_idx,
                           thresholds=np.arange(1.0, 2.5, 0.05),
                           ITI=0.5, n_repeats=100):
    """
    Finds LCA threshold z that maximizes reward rate: acc / (ITI + RT)
    For use inside a run_prp_trial() after full integration.
    """
    best_rr = -np.inf
    best_threshold = None

    for z in thresholds:
        rt, choice = run_lca_avg(
            input_series=input_series,
            relevant_output_indices=relevant_output_indices,
            threshold=z,
            n_repeats=n_repeats
        )

        if rt is None:
            continue

        acc = int(choice == correct_response_idx)
        rr = acc / (ITI + rt)

        if rr > best_rr:
            best_rr = rr
            best_threshold = z

    return best_threshold


def optimize_reward_rate_threshold(net, input_a, input_b, task_a, task_b,
                                   soa, max_timesteps=100,
                                   thresholds=np.arange(1.0, 2.5, 0.05),
                                   tau_net=0.2, tau_task=0.2, persistence=0.5,
                                   ITI=0.5):
    """
    Runs a simulated PRP trial and finds the threshold z that maximizes reward rate.
    Used in sweep_soa(), BEFORE running full trials.
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
        persistence=persistence
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

    # Try each threshold
    best_rr = -np.inf
    best_z = None
    for z in thresholds:
        rt, choice = run_lca_avg(
            input_series=output_series,
            relevant_output_indices=output_idxs,
            threshold=z,
            n_repeats=100
        )
        if rt is None:
            continue

        acc = int(choice == correct_idx)
        rr = acc / (ITI + rt)

        if rr > best_rr:
            best_rr = rr
            best_z = z

    return best_z
