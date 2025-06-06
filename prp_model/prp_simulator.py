import numpy as np
from prp_model.lca import run_lca
from prp_model.lca import run_lca_avg


def run_prp_trial(task_net, input_a, input_b, task_a, task_b,
                  soa, max_timesteps=100, threshold=1.0,
                  tau_net=0.2, tau_task=0.2, persistence=0.5):
    """
    Continuous PRP trial: Task A starts at t=0, Task B starts at t=SOA.
    Both are processed in parallel, with interference emerging from overlap.
    """
    input_dim = input_a.shape[0]
    task_dim = task_a.shape[0]
    N_pathways = 3
    N_features = 3

    input_series = []
    task_series = []

    for t in range(max_timesteps):
        stim_t = np.zeros(input_dim)
        task_t = np.zeros(task_dim)

        if t >= 0:
            stim_t += input_a
            task_t += task_a

        if t >= soa:
            stim_t += input_b
            task_t += task_b  # both tasks are active after SOA

        input_series.append(stim_t)
        task_series.append(task_t)

    # Run forward pass with both tasks present
    output_series = task_net.integrate(input_series, task_series,
                                       tau_net=tau_net, tau_task=tau_task, 
                                       persistence=persistence)

    # --- Task A RT ---
    task_matrix_a = task_a.reshape(N_pathways, N_pathways).T
    in_dim_a, out_dim_a = np.argwhere(task_matrix_a == 1)[0]
    relevant_outputs_a = list(range(out_dim_a * N_features, (out_dim_a + 1) * N_features))

    rt_a, choice_a, _ = run_lca(
        input_series=output_series,
        relevant_output_indices=relevant_outputs_a,
        threshold=threshold,
        max_timesteps=max_timesteps
    )
    acc_a = (choice_a == np.argmax(input_a[in_dim_a * N_features : (in_dim_a + 1) * N_features]))

    # --- Task B RT ---
    task_matrix_b = task_b.reshape(N_pathways, N_pathways).T
    in_dim_b, out_dim_b = np.argwhere(task_matrix_b == 1)[0]
    relevant_outputs_b = list(range(out_dim_b * N_features, (out_dim_b + 1) * N_features))

    # !!!Changed run_lca to run_lca_avg. Do the same for Task A later on!!!
    rt_b, choice_b = run_lca_avg(
        input_series=output_series[soa:],
        relevant_output_indices=relevant_outputs_b,
        n_repeats=100,
        threshold=threshold,
        max_timesteps=max_timesteps - soa
    )

    if rt_b is not None:
        rt_b += soa * 0.1  # convert back to full timeline

    acc_b = (choice_b == np.argmax(input_b[in_dim_b * N_features : (in_dim_b + 1) * N_features]))

    return rt_a, acc_a, rt_b, acc_b, output_series

# Sweep SOA function simulating various SOAs
def sweep_soa(task_net,
              trial_generator,
              soa_values,
              n_trials_per_soa=10, # Figure out correct number of trials
              threshold=1.0, # Figure out correct threshold value
              max_timesteps=50,
              tau_net=0.2, # 0.2
              tau_task=0.2, # 0.2
              persistence=0.0, # Paper: 0.0, 0.5, 0.8, 0.9
              verbose=False):
    """
    Runs PRP simulation across a range of SOAs using the updated continuous input version.
    Each trial presents Task A at t=0 and Task B at t=SOA.
    """

    results = {
        "soa": [],
        "rt_a": [],
        "rt_b": [],
        "acc_a": [],
        "acc_b": []
    }

    for soa in soa_values:
        rt_a_list, rt_b_list = [], []
        acc_a_list, acc_b_list = [], []

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
                threshold=threshold,
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
            print(f"SOA {soa} | RT_A: {results['rt_a'][-1]:.2f} | RT_B: {results['rt_b'][-1]:.2f}")

    return results


