# prp/prp_simulator.py

import numpy as np
import torch
from prp.lca import run_lca_avg
from prp.threshold_utils import optimize_lca_threshold_dist

DEFAULT_N_REPEATS = 100  # number of repeats for LCA simulations

def run_prp_trial(
    task_net,
    input_a, input_b,
    task_a, task_b,
    soa: int,
    max_timesteps: int = 100,
    persistence: float = 0.5,
    thresholds: np.ndarray = np.arange(0.0, 1.6, 0.1),
    ITI: float = 4.0,
    n_repeats: int = DEFAULT_N_REPEATS,
    z_b_fixed: float = None
):
    """
    Runs a single PRP trial with Task A at t=0 and Task B at t=soa (fixed).
    Returns (rt_a, acc_a, rt_b, acc_b, output_np), where output_np is
    the network activation time-series as a NumPy array of shape (T, D_out).
    """
    # 1) Task B onset fixed to the experimental SOA
    onset_b = soa

    # 2) Build the raw time-series as Python lists
    input_series = []
    task_series  = []
    input_dim = input_a.shape[0]
    task_dim  = task_a.shape[0]

    for t in range(max_timesteps):
        stim_t = np.zeros(input_dim, dtype=np.float32)
        task_t = np.zeros(task_dim,  dtype=np.float32)

        # Task A present from t >= 0
        stim_t += input_a
        task_t += task_a

        # Task B present from t >= onset_b
        if t >= onset_b:
            stim_t += input_b
            task_t += task_b

        input_series.append(stim_t)
        task_series.append(task_t)

    # 3a) Stack into NumPy arrays once
    input_np = np.stack(input_series, axis=0)  # shape (T, input_dim)
    task_np  = np.stack(task_series,  axis=0)  # shape (T, task_dim)

    # 3b) Convert to torch tensors for integration
    input_th = torch.from_numpy(input_np)
    task_th  = torch.from_numpy(task_np)

    # 3c) Run network integration
    output_series_th = task_net.integrate(
        input_th,
        task_th,
        persistence=persistence
    )

    # 3d) Convert back to NumPy array for LCA
    output_np = np.stack([o.numpy() for o in output_series_th], axis=0)

    # helper: decode which indices and correct response for a task
    def _decode(task_vec, input_vec, N_pathways=3, N_features=3):
        mat = task_vec.reshape(N_pathways, N_pathways).T
        in_dim, out_dim = np.argwhere(mat == 1)[0]
        # correct feature within the input
        correct = np.argmax(input_vec[in_dim * N_features:(in_dim+1)*N_features])
        idxs = list(range(out_dim * N_features, (out_dim+1)*N_features))
        return idxs, correct

    # 4) LCA for Task A (entire series)
    idxs_a, corr_a = _decode(task_a, input_a)
    z_a, _ = optimize_lca_threshold_dist(
        output_np, idxs_a,
        correct_response_idx=corr_a,
        thresholds=thresholds,
        ITI=ITI,
        n_repeats=n_repeats
    )
    rt_a, choice_a = run_lca_avg(
        output_np, idxs_a,
        threshold=z_a,
        n_repeats=n_repeats
    )
    acc_a = (choice_a == corr_a) if rt_a is not None else False

    # 5) LCA for Task B (only tail after onset)
    idxs_b, corr_b = _decode(task_b, input_b)
    tail = output_np[onset_b:]
    # pick z_b_fixed if given, else optimize
    if z_b_fixed is None:
        z_b, _ = optimize_lca_threshold_dist(
            tail, idxs_b,
            correct_response_idx=corr_b,
            thresholds=thresholds,
            ITI=ITI,
            n_repeats=n_repeats
        )
    else:
        z_b = z_b_fixed

    rt_b, choice_b = run_lca_avg(
        tail, idxs_b,
        threshold=z_b,
        n_repeats=n_repeats
    )
    # bring RT_B back to absolute time
    if rt_b is not None:
        rt_b = rt_b + onset_b
    acc_b = (choice_b == corr_b) if rt_b is not None else False

    return rt_a, acc_a, rt_b, acc_b, output_np


def sweep_soa(
    task_net,
    trial_generator,
    soa_values,
    n_trials_per_soa: int = 10,
    max_timesteps: int = 50,
    persistence: float = 0.5,
    n_repeats: int = DEFAULT_N_REPEATS,
    verbose: bool = False,
    z_b_fixed: float = None
):
    """
    Runs PRP simulations across a list of SOAs.
    Returns a dict with keys: soa, rt_a, acc_a, rt_b, acc_b.
    """
    results = {k: [] for k in ("soa", "rt_a", "acc_a", "rt_b", "acc_b")}

    for soa in soa_values:
        rt_a_vals, acc_a_vals = [], []
        rt_b_vals, acc_b_vals = [], []

        if verbose:
            print(f"▶️ Starting SOA = {soa}")

        for _ in range(n_trials_per_soa):
            inp_a, inp_b, t_a, t_b = trial_generator()

            rt_a, acc_a, rt_b, acc_b, _ = run_prp_trial(
                task_net=task_net,
                input_a=inp_a,
                input_b=inp_b,
                task_a=t_a,
                task_b=t_b,
                soa=soa,
                max_timesteps=max_timesteps,
                persistence=persistence,
                n_repeats=n_repeats,
                z_b_fixed=z_b_fixed
            )

            # collect only valid, non-None RTs
            if rt_a is not None and not np.isnan(rt_a):
                rt_a_vals.append(rt_a)
                acc_a_vals.append(acc_a)
            if rt_b is not None and not np.isnan(rt_b):
                rt_b_vals.append(rt_b)
                acc_b_vals.append(acc_b)

        # aggregate
        results["soa"].append(soa)
        results["rt_a"].append(np.mean(rt_a_vals) if rt_a_vals else np.nan)
        results["acc_a"].append(np.mean(acc_a_vals) if acc_a_vals else np.nan)
        results["rt_b"].append(np.mean(rt_b_vals) if rt_b_vals else np.nan)
        results["acc_b"].append(np.mean(acc_b_vals) if acc_b_vals else np.nan)

        if verbose:
            print(
                f"✅ SOA={soa} | "
                f"A: RT={results['rt_a'][-1]:.2f}, ACC={results['acc_a'][-1]:.2f} | "
                f"B: RT={results['rt_b'][-1]:.2f}, ACC={results['acc_b'][-1]:.2f}"
            )

    return results
