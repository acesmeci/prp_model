import numpy as np
from prp.lca import run_lca, run_lca_avg
from prp.choose_onset_policy import choose_onset_policy
from prp.threshold_optimizer import (
    optimize_lca_threshold,
    optimize_reward_rate_threshold
)


def run_prp_trial(task_net, input_a, input_b, task_a, task_b,
                  soa, max_timesteps=100,
                  tau_net=0.2, tau_task=0.2, persistence=0.5):
    """
    Continuous PRP trial: Task A starts at t=0, Task B starts at t=onset_b (≥ SOA).
    Task A onset is optimized for reward rate. Threshold is optimized per task.
    """
    input_dim = input_a.shape[0]
    task_dim = task_a.shape[0]
    N_pathways = 3
    N_features = 3

    input_series = []
    task_series = []

    onset_b = choose_onset_policy(
        task_net=task_net,
        input_a=input_a,
        input_b=input_b,
        task_a=task_a,
        task_b=task_b,
        soa=soa,
        tau_net=tau_net,
        tau_task=tau_task,
        persistence=persistence
    )

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

    output_series = task_net.integrate(
        input_series, task_series,
        tau_net=tau_net,
        tau_task=tau_task,
        persistence=persistence
    )

    # Task A
    task_matrix_a = task_a.reshape(N_pathways, N_pathways).T
    in_dim_a, out_dim_a = np.argwhere(task_matrix_a == 1)[0]
    output_idxs_a = list(range(out_dim_a * N_features, (out_dim_a + 1) * N_features))
    correct_a = np.argmax(input_a[in_dim_a * N_features:(in_dim_a + 1) * N_features])

    z_a = optimize_lca_threshold(output_series, output_idxs_a, correct_response_idx=correct_a)
    rt_a, choice_a, _ = run_lca(output_series, output_idxs_a, threshold=z_a, max_timesteps=max_timesteps)
    acc_a = (choice_a == correct_a)

    # Task B
    task_matrix_b = task_b.reshape(N_pathways, N_pathways).T
    in_dim_b, out_dim_b = np.argwhere(task_matrix_b == 1)[0]
    output_idxs_b = list(range(out_dim_b * N_features, (out_dim_b + 1) * N_features))
    correct_b = np.argmax(input_b[in_dim_b * N_features:(in_dim_b + 1) * N_features])

    z_b = optimize_lca_threshold(output_series[onset_b:], output_idxs_b, correct_response_idx=correct_b)
    rt_b, choice_b = run_lca_avg(
        input_series=output_series[onset_b:],
        relevant_output_indices=output_idxs_b,
        threshold=z_b,
        n_repeats=100,
        max_timesteps=max_timesteps - onset_b
    )

    if rt_b is not None:
        rt_b += onset_b * 0.1

    acc_b = (choice_b == correct_b)

    return rt_a, acc_a, rt_b, acc_b, output_series


def sweep_soa(task_net,
              trial_generator,
              soa_values,
              n_trials_per_soa=10,
              max_timesteps=50,
              tau_net=0.2,
              tau_task=0.2,
              persistence=0.0,
              verbose=False):
    """
    Runs PRP simulation across SOAs using reward-rate-optimized thresholds.
    """
    results = {
        "soa": [],
        "rt_a": [],
        "rt_b": [],
        "acc_a": [],
        "acc_b": [],
        "threshold": []
    }

    for soa in soa_values:
        rt_a_list, rt_b_list = [], []
        acc_a_list, acc_b_list = [], []

        input_a, input_b, task_a, task_b = trial_generator()
        optimized_thresh = optimize_reward_rate_threshold(
            net=task_net,
            input_a=input_a,
            input_b=input_b,
            task_a=task_a,
            task_b=task_b,
            soa=soa,
            tau_net=tau_net,
            tau_task=tau_task,
            persistence=persistence
        )

        results["threshold"].append(optimized_thresh)
        if verbose:
            print(f"✅ SOA {soa}: Optimized threshold = {optimized_thresh:.2f}")

        for _ in range(n_trials_per_soa):
            input_a, input_b, task_a, task_b = trial_generator()

            rt_a, acc_a, rt_b, acc_b, _ = run_prp_trial(
                task_net=task_net,
                input_a=input_a,
                input_b=input_b,
                task_a=task_a,
                task_b=task_b,
                soa=soa,
                max_timesteps=max_timesteps,
                tau_net=tau_net,
                tau_task=tau_task,
                persistence=persistence
            )

            if rt_a is not None:
                rt_a_list.append(rt_a)
                acc_a_list.append(acc_a)

            if rt_b is not None:
                rt_b_list.append(rt_b)
                acc_b_list.append(acc_b)

        results["soa"].append(soa)
        results["rt_a"].append(np.mean(rt_a_list) if rt_a_list else np.nan)
        results["rt_b"].append(np.mean(rt_b_list) if rt_b_list else np.nan)
        results["acc_a"].append(np.mean(acc_a_list) if acc_a_list else np.nan)
        results["acc_b"].append(np.mean(acc_b_list) if acc_b_list else np.nan)

        if verbose:
            print(f"RT_A: {results['rt_a'][-1]:.2f} | RT_B: {results['rt_b'][-1]:.2f} | ACC_A: {results['acc_a'][-1]:.2f} | ACC_B: {results['acc_b'][-1]:.2f}")

    return results
