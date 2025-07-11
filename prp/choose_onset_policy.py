import numpy as np
from prp.lca import run_lca_avg  
from prp.threshold_optimizer import optimize_threshold  # ✅ NEW

def choose_onset_policy(task_net, input_a, input_b, task_a, task_b,
                        max_onset_delay=15, soa=3,
                        n_repeats=20,
                        tau_net=0.2, tau_task=0.2, persistence=0.5,
                        ITI=0.5):
    """
    Searches for Task 2 onset that maximizes joint reward rate.
    
    Returns:
        optimal_onset (int): time step to start Task 2
    """
    best_rr = -np.inf
    best_onset = 0
    N_pathways = 3
    N_features = 3

    # Determine output indices
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

        for t in range(delay + 50):  # simulate long enough to finish both
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
        z_a = optimize_threshold(output_series, output_a, correct_response_idx=correct_a, ITI=ITI, n_repeats=n_repeats)
        z_b = optimize_threshold(output_series[delay:], output_b, correct_response_idx=correct_b, ITI=ITI, n_repeats=n_repeats)

        # === Evaluate performance ===
        rt_a, choice_a = run_lca_avg(output_series, output_a, threshold=z_a, n_repeats=n_repeats)
        acc_a = int(choice_a == correct_a) if choice_a is not None else 0

        rt_b, choice_b = run_lca_avg(output_series[delay:], output_b, threshold=z_b, n_repeats=n_repeats)
        acc_b = int(choice_b == correct_b) if choice_b is not None else 0

        if rt_a is None or rt_b is None:
            continue

        rt_total = max(rt_a, delay * 0.1 + rt_b)
        reward_rate = (acc_a * acc_b) / (ITI + rt_total)

        if reward_rate > best_rr:
            best_rr = reward_rate
            best_onset = delay

    return best_onset
