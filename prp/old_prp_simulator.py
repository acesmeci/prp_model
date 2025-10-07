# prp/old_prp_simulator.py

import numpy as np
import torch
from prp.lca import run_lca_avg
from prp.threshold_utils import optimize_lca_threshold_dist, choose_onset_policy

DEFAULT_N_REPEATS = 100  # Number of LCA simulations per trial for averaging


def run_prp_trial(
    task_net,
    input_a, input_b,
    task_a, task_b,
    soa: int,
    max_timesteps: int = 100,
    persistence: float = 0.5,
    thresholds: np.ndarray = np.arange(0.0, 1.6, 0.1),
    ITI: float = 0.5,
    n_repeats: int = DEFAULT_N_REPEATS,
    z_b_fixed: float = None,
    dt_lca: float = 0.1,
    t0: float = 0.15,
    optimize_onset: bool = True,
    policy_n_repeats: int = 30,
    thresholds_policy: np.ndarray | None = None,
    max_onset_delay: int = 5,
):
    """
    Returns (rt_a, acc_a, rt_b, acc_b, output_np)
    output_np is the final (gated) pass output time-series.
    """

    def _decode(task_vec, input_vec, N_pathways=3, N_features=3):
        mat = task_vec.reshape(N_pathways, N_pathways) # removed .T here, check if correct
        in_dim, out_dim = np.argwhere(mat == 1)[0]
        correct = np.argmax(input_vec[in_dim*N_features:(in_dim+1)*N_features])
        idxs = list(range(out_dim*N_features, (out_dim+1)*N_features))
        return idxs, correct

    def _integrate(input_series, task_series):
        input_np = np.stack(input_series, axis=0).astype(np.float32)
        task_np  = np.stack(task_series,  axis=0).astype(np.float32)
        out_th   = task_net.integrate(torch.from_numpy(input_np),
                                      torch.from_numpy(task_np),
                                      persistence=persistence)
        return np.stack([o.numpy() for o in out_th], axis=0)

    # 0) Decide Task-B onset
    if optimize_onset:
        onset_b = choose_onset_policy(
            task_net, input_a, input_b, task_a, task_b,
            soa=soa,
            max_onset_delay=max_onset_delay,
            max_timesteps=max_timesteps,
            persistence=persistence,
            ITI=ITI,
            dt_lca=dt_lca,
            t0=t0,
            z_b_fixed=z_b_fixed,              # reuse fixed B threshold for speed
            policy_n_repeats=policy_n_repeats,
            thresholds_policy=thresholds_policy
        )
    else:
        onset_b = soa

    # 1) Pass 1: A on (B from onset) -> A's RT
    input_series, task_series = [], []
    inp_dim, task_dim = input_a.shape[0], task_a.shape[0]
    for t in range(max_timesteps):
        stim_t = np.zeros(inp_dim, dtype=np.float32)
        task_t = np.zeros(task_dim, dtype=np.float32)
        stim_t += input_a
        if t >= onset_b:
            stim_t += input_b
        task_t += task_a
        if t >= onset_b:
            task_t += task_b
        input_series.append(stim_t)
        task_series.append(task_t)
    out1 = _integrate(input_series, task_series)

    idxs_a, corr_a = _decode(task_a, input_a)
    z_a, _ = optimize_lca_threshold_dist(
        out1, idxs_a,
        correct_response_idx=corr_a,
        thresholds=thresholds,
        ITI=ITI,
        n_repeats=n_repeats
    )
    rt_a, choice_a = run_lca_avg(out1, idxs_a, threshold=z_a,
                                 n_repeats=n_repeats, dt=dt_lca)
    acc_a = (choice_a == corr_a) if rt_a is not None else False
    t_off_a = int(np.ceil(max(0.0, (rt_a - t0) / dt_lca))) if rt_a is not None else max_timesteps
    # debugging to verify if overlap ends by ~1s
    print(f"SOA={soa}  onset_b={onset_b} ({onset_b*dt_lca:.2f}s)  "
      f"t_off_a={t_off_a} ({t_off_a*dt_lca:.2f}s)  "
      f"overlap≈{max(0, t_off_a - onset_b)*dt_lca:.2f}s")


    # 2) Pass 2: turn OFF A's task unit after its decision -> evaluate B on tail
    input_series, task_series = [], []
    for t in range(max_timesteps):
        stim_t = np.zeros(inp_dim, dtype=np.float32)
        task_t = np.zeros(task_dim, dtype=np.float32)
        stim_t += input_a
        if t >= onset_b:
            stim_t += input_b
        if t < t_off_a:
            task_t += task_a
        if t >= onset_b:
            task_t += task_b
        input_series.append(stim_t)
        task_series.append(task_t)
    out2 = _integrate(input_series, task_series)

    idxs_b, corr_b = _decode(task_b, input_b)
    tail = out2[onset_b:]
    if tail.shape[0] == 0:
        return rt_a, acc_a, None, False, out2

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

    rt_b, choice_b = run_lca_avg(tail, idxs_b, threshold=z_b,
                                 n_repeats=n_repeats, dt=dt_lca)
    if rt_b is not None:
        rt_b = rt_b + onset_b * dt_lca
    acc_b = (choice_b == corr_b) if rt_b is not None else False

    return rt_a, acc_a, rt_b, acc_b, out2

def sweep_soa(
    task_net,
    trial_generator,
    soa_values,
    n_trials_per_soa: int = 10,
    max_timesteps: int = 100,
    persistence: float = 0.5,
    n_repeats: int = DEFAULT_N_REPEATS,
    verbose: bool = False,
    z_b_fixed: float = None,
    dt_lca: float = 0.1,
    t0: float = 0.15,
    ITI: float = 0.5,
    optimize_onset: bool = True,
    thresholds: np.ndarray = np.arange(0.0, 1.6, 0.1),
):
    """
    Runs PRP simulations across SOAs.
    Returns: dict with keys: soa, rt_a, acc_a, rt_b, acc_b
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
                thresholds=thresholds,
                ITI=ITI,
                n_repeats=n_repeats,
                z_b_fixed=z_b_fixed,
                dt_lca=dt_lca,
                t0=t0,
                optimize_onset=optimize_onset,
            )

            if rt_a is not None and not np.isnan(rt_a):
                rt_a_vals.append(rt_a); acc_a_vals.append(acc_a)
            if rt_b is not None and not np.isnan(rt_b):
                rt_b_vals.append(rt_b); acc_b_vals.append(acc_b)

        results["soa"].append(soa)
        results["rt_a"].append(np.mean(rt_a_vals) if rt_a_vals else np.nan)
        results["acc_a"].append(np.mean(acc_a_vals) if acc_a_vals else np.nan)
        results["rt_b"].append(np.mean(rt_b_vals) if rt_b_vals else np.nan)
        results["acc_b"].append(np.mean(acc_b_vals) if acc_b_vals else np.nan)

        if verbose:
            print(
                f"✅ SOA={soa} | "
                f"A: RT={results['rt_a'][-1]:.3f}, ACC={results['acc_a'][-1]:.2f} | "
                f"B: RT={results['rt_b'][-1]:.3f}, ACC={results['acc_b'][-1]:.2f}"
            )

    return results
