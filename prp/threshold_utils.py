# Legacy threshold optimizers (currently unused, kept for fallback/debug):
# - optimize_reward_rate_threshold
# - optimize_lca_threshold

import numpy as np
from prp.lca import run_lca_avg
from prp.lca import run_lca_dist
import torch

DEFAULT_N_REPEATS = 100 # Default number of repeats for LCA simulations. Use 100 if you have access to a GPU.

# Not used in the current implementation. Replaced by optimize_lca_threshold_dist.
def optimize_lca_threshold(input_series, relevant_output_indices, correct_response_idx,
                           thresholds=np.arange(0.0, 1.6, 0.1),
                           ITI=0.5,
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


# Optimize LCA threshold with run_lca_dist. Faithful to MATLAB implementation
# Correct_response_idx is not used directly here, because run_lca_dist() infers correctness from argmax(p[0]). Change this if needed.
def optimize_lca_threshold_dist(
    input_series,
    relevant_output_indices,
    correct_response_idx=None,
    thresholds=np.arange(0.1, 1.6, 0.1),
    ITI=0.5,
    n_repeats=100,
    dt=0.1,
    tau=0.1,
    lambda_=0.4,
    alpha=0.2,
    beta=0.2,
    noise_std=0.2,
    t0=0.15,
    verbose=False,
):
    """
    Sweep thresholds; choose the one with the highest reward-rate.
    Returns (best_threshold, results_dict).
    """
    results = run_lca_dist(
        input_series=input_series,
        relevant_output_indices=relevant_output_indices,
        thresholds=thresholds,
        n_repeats=n_repeats,
        dt=dt, tau=tau, lambda_=lambda_, alpha=alpha, beta=beta,
        noise_std=noise_std, t0=t0, ITI=ITI,
        correct_response_idx=correct_response_idx,
    )

    rr = results["reward_rates"]
    best_idx = int(np.argmax(rr))             # safe: RR invalids were set to 0
    best_threshold = float(results["thresholds"][best_idx])

    if verbose:
        accs, rts, zs = results["accuracies"], results["rts"], results["thresholds"]
        for i in range(len(zs)):
            print(f"z={zs[i]:.2f} | Acc={accs[i]:.2f} | RT={rts[i]:.3f} | RR={rr[i]:.3f}")
        print(f"✅ Best threshold z: {best_threshold:.2f}")

    return best_threshold, results



# Updated version of choose_onset_policy to use optimize_lca_threshold_dist
# Can add verbose if needed
def choose_onset_policy(
    task_net,
    input_a, input_b,
    task_a, task_b,
    soa: int = 0,
    max_onset_delay: int = 5,      # small, fast search window
    max_timesteps: int = 100,
    persistence: float = 0.5,
    ITI: float = 0.5,
    dt_lca: float = 0.1,
    t0: float = 0.15,
    # speedups for policy search:
    z_a_fixed: float | None = None,
    z_b_fixed: float | None = None,
    policy_n_repeats: int = 30,
    thresholds_policy: np.ndarray | None = None,  # coarse grid for policy search
):
    """
    Returns best onset (int, in steps) that maximizes:
        RR = (Acc_A * Acc_B) / (ITI + max(RT_A, RT_B_abs))

    Two-pass integration per candidate onset:
      Pass1: A task unit ON (B from onset) -> get RT_A, gate time for A
      Pass2: Turn OFF A's task unit after its decision -> evaluate B on the tail
    """
    if thresholds_policy is None:
        thresholds_policy = np.linspace(0.1, 0.6, 6) # should be 0.1 -> 1.6 for faithfulness, but faster for policy search

    def _decode(task_vec, input_vec, N_pathways=3, N_features=3):
        mat = task_vec.reshape(N_pathways, N_pathways) # removed .T here, check if correct
        in_dim, out_dim = np.argwhere(mat == 1)[0]
        correct = np.argmax(input_vec[in_dim*N_features:(in_dim+1)*N_features])
        idxs = list(range(out_dim*N_features, (out_dim+1)*N_features))
        return idxs, correct

    def _integrate_once(onset_b: int, gate_after_a_steps: int | None):
        inp_dim, task_dim = input_a.shape[0], task_a.shape[0]
        input_series, task_series = [], []
        for t in range(max_timesteps):
            stim_t = np.zeros(inp_dim, dtype=np.float32)
            task_t = np.zeros(task_dim, dtype=np.float32)
            # stimuli
            stim_t += input_a
            if t >= onset_b:
                stim_t += input_b
            # task units (control)
            if gate_after_a_steps is None or t < gate_after_a_steps:
                task_t += task_a
            if t >= onset_b:
                task_t += task_b
            input_series.append(stim_t)
            task_series.append(task_t)
        input_np = np.stack(input_series, axis=0).astype(np.float32)
        task_np  = np.stack(task_series,  axis=0).astype(np.float32)
        out_th = task_net.integrate(torch.from_numpy(input_np),
                                    torch.from_numpy(task_np),
                                    persistence=persistence)
        return np.stack([o.numpy() for o in out_th], axis=0)  # [T, D_out]

    # decode once
    idxs_a, corr_a = _decode(task_a, input_a)
    idxs_b, corr_b = _decode(task_b, input_b)

    best_rr = -np.inf
    best_onset = int(soa)
    onset_upper = min(int(soa) + int(max_onset_delay), max_timesteps - 1)

    for onset in range(int(soa), onset_upper + 1):
        # ---- Pass 1: A timing ----
        out1 = _integrate_once(onset_b=onset, gate_after_a_steps=None)
        if z_a_fixed is None:
            z_a, _ = optimize_lca_threshold_dist(
                out1, idxs_a,
                correct_response_idx=corr_a,
                thresholds=thresholds_policy,
                ITI=ITI,
                n_repeats=policy_n_repeats
            )
        else:
            z_a = z_a_fixed
        rt_a, choice_a = run_lca_avg(out1, idxs_a, threshold=z_a,
                                     n_repeats=policy_n_repeats, dt=dt_lca)
        if rt_a is None:
            continue
        acc_a = 1.0 if choice_a == corr_a else 0.0
        t_off_a = int(np.ceil(max(0.0, (rt_a - t0) / dt_lca)))

        # ---- Pass 2: gate A then evaluate B ----
        out2 = _integrate_once(onset_b=onset, gate_after_a_steps=t_off_a)
        tail = out2[onset:]
        if tail.shape[0] == 0:
            continue

        if z_b_fixed is None:
            z_b, _ = optimize_lca_threshold_dist(
                tail, idxs_b,
                correct_response_idx=corr_b,
                thresholds=thresholds_policy,
                ITI=ITI,
                n_repeats=policy_n_repeats
            )
        else:
            z_b = z_b_fixed
        rt_b, choice_b = run_lca_avg(tail, idxs_b, threshold=z_b,
                                     n_repeats=policy_n_repeats, dt=dt_lca)
        if rt_b is None:
            continue
        acc_b = 1.0 if choice_b == corr_b else 0.0
        rt_b_abs = rt_b + onset * dt_lca

        rr = (acc_a * acc_b) / (ITI + max(rt_a, rt_b_abs))
        if rr > best_rr:
            best_rr = rr
            best_onset = onset

    return int(best_onset)



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
