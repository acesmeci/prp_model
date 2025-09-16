# prp/task_generator.py
# !!!Legacy now!!! Use training_set.py instead.
"""
Task/stimulus generator for the PRP simulations.

This module creates single–task training/evaluation datasets for five canonical
tasks over 3 input/output pathways with N_features per pathway. Each stimulus
has exactly one active feature per input pathway (one-hot within the pathway).
A task cue (one-hot over N_pathways^2 possible mappings) selects a mapping
from one input pathway to one output pathway (e.g., A: in0→out0, D: in0→out1).

Two entry points:
- generate_task_patterns: generic generator (historical); supports arbitrary
  task IDs but currently used as single-task builder.
- generate_fixed_task_set: convenience wrapper that builds the five tasks
  A–E used throughout the paper and these simulations, and returns metadata.

All generators add small Gaussian noise to input features (per-pathway,
diagonal covariance) to avoid degenerate, trivially separable data.
"""

import numpy as np

def generate_task_patterns(
    N_pathways,
    N_features,
    samples_per_task=None,
    sd_scale=0.25,
    same_stimuli_across_tasks=False,
    relevant_tasks=None,
    generate_multitasking_patterns=True,
    zero_dimensions=False,
    seed=None
):
    """
    Build single-task examples for a user-specified subset of the N_pathways^2
    possible tasks (input→output mappings).

    Parameters
    ----------
    N_pathways : int
        Number of input/output pathways (typically 3).
    N_features : int
        Number of discrete features per pathway (typically 3).
    samples_per_task : int or None
        If None, generate all combinations of features (N_features^N_pathways).
        If int, sample that many random feature combinations per task.
    sd_scale : float
        Scales the diagonal Gaussian noise added to each pathway's feature
        vector. A per-feature variance row is drawn uniformly in [0, sd_scale).
    same_stimuli_across_tasks : bool
        Reserved; not currently used. (If enabled, would reuse the same set
        of stimuli across different task cues.)
    relevant_tasks : list[int] or None
        1-indexed IDs of tasks to include (MATLAB-style). Defaults to all tasks.
    generate_multitasking_patterns : bool
        Reserved; not currently used in this function.
    zero_dimensions : bool
        If True, drops the last feature of each pathway (for legacy cases with
        a dummy feature).
    seed : int or None
        Seed for NumPy RNG (local to this function).

    Returns
    -------
    input_sgl : (N, N_pathways*N_features) float32
        Noisy input stimuli. Each pathway block has one dominant feature.
    tasks_sgl : (N, N_pathways^2) float32
        One-hot task cues (exactly one active unit specifies input→output).
    train_sgl : (N, N_pathways*N_features) float32
        One-hot targets. For task (i→o), the active feature on input pathway i
        is copied to output pathway o.
    meta : dict
        {
          "input_sgl_mask": binary (noise-free) inputs,
          "tasks_idx": 1-indexed task IDs per row,
          "stim_idx": per-task stimulus index per row,
        }

    Notes
    -----
    • Noise model: for each pathway p, we draw from N(μ, Σ_p) where μ is the
      one-hot feature vector and Σ_p = diag(sd_row), with sd_row selected by the
      active feature index (row of a random N_features×N_features matrix scaled
      by sd_scale). This adds within-pathway jitter but keeps features local.
    • Targets are exact functions of (input, task), independent of noise.
    """

    if seed is not None:
        np.random.seed(seed)

    I = N_pathways * N_features  # total input units
    T = N_pathways ** 2          # total possible tasks
    R = I                        # total output units (same dimensionality)

    # define task subset
    if relevant_tasks is None:
        relevant_tasks = list(range(1, T + 1))  # 1-indexed like in MATLAB

    stim_mask_list = []
    task_mask_list = []
    train_mask_list = []
    task_idx_list = []
    stim_idx_list = []

    for t_idx, task_id in enumerate(relevant_tasks):
        task_vector = np.zeros(T)
        task_vector[task_id - 1] = 1

        if samples_per_task is None:
            # Generate all combinations
            feature_combs = np.array(np.meshgrid(*[range(N_features) for _ in range(N_pathways)])).T.reshape(-1, N_pathways)
        else:
            feature_combs = np.random.randint(0, N_features, size=(samples_per_task, N_pathways))

        curr_input_mask = np.zeros((feature_combs.shape[0], I))
        curr_output_mask = np.zeros((feature_combs.shape[0], R))

        for i, feat_idx in enumerate(feature_combs):
            for d in range(N_pathways):
                curr_input_mask[i, d * N_features + feat_idx[d]] = 1

        # Convert task_id to matrix form to identify input-output mapping
        curr_tasksM = task_vector.reshape(N_pathways, N_pathways) # removed .T here, check if correct
        input_dim, output_dim = np.argwhere(curr_tasksM == 1)[0]  # only one active task

        for i in range(curr_input_mask.shape[0]):
            start_in = input_dim * N_features
            start_out = output_dim * N_features
            curr_output_mask[i, start_out:start_out + N_features] = curr_input_mask[i, start_in:start_in + N_features]

        stim_mask_list.append(curr_input_mask)
        task_mask_list.append(np.tile(task_vector, (curr_input_mask.shape[0], 1)))
        train_mask_list.append(curr_output_mask)
        task_idx_list.append(np.full((curr_input_mask.shape[0],), task_id))
        stim_idx_list.append(np.arange(1, curr_input_mask.shape[0] + 1))

    input_sgl_mask = np.vstack(stim_mask_list)
    tasks_sgl_mask = np.vstack(task_mask_list)
    train_sgl_mask = np.vstack(train_mask_list)
    tasks_idx_sgl = np.concatenate(task_idx_list)
    stim_idx_sgl = np.concatenate(stim_idx_list)

    # Gaussian noise injection
    input_sgl = np.zeros_like(input_sgl_mask)
    sd_matrix = np.random.rand(N_features, N_features) * sd_scale

    for row_idx in range(input_sgl_mask.shape[0]):
        for dim in range(N_pathways):
            col_start = dim * N_features
            col_end = (dim + 1) * N_features
            mu = input_sgl_mask[row_idx, col_start:col_end]
            active_feat = np.argmax(mu)
            input_sgl[row_idx, col_start:col_end] = np.random.multivariate_normal(mu, np.diag(sd_matrix[active_feat]))

    # Optionally remove zero-dimensions (extra dummy feature)
    if zero_dimensions:
        zero_idx = [(i + 1) * N_features - 1 for i in range(N_pathways)]
        input_sgl = np.delete(input_sgl, zero_idx, axis=1)
        input_sgl_mask = np.delete(input_sgl_mask, zero_idx, axis=1)
        train_sgl_mask = np.delete(train_sgl_mask, zero_idx, axis=1)

    return input_sgl, tasks_sgl_mask, train_sgl_mask, {
        "input_sgl_mask": input_sgl_mask,
        "tasks_idx": tasks_idx_sgl,
        "stim_idx": stim_idx_sgl
        # multitask support: add later as needed
    }



# Generate fixed task patterns

# prp/task_generator.py
def generate_fixed_task_set(
    N_pathways: int = 3,
    N_features: int = 3,
    samples_per_task: int = 100,
    sd_scale: float = 0.25,
    seed: int = None
):
    """
    Build the canonical five tasks (A–E) used in the paper:

        A: in0 → out0
        B: in1 → out1
        C: in2 → out2
        D: in0 → out1   (shares input with A; output with B)
        E: in1 → out0   (shares input with B; output with A)

    Each stimulus contains one active feature per input pathway (one-hot within
    pathway). For task (i→o), the target copies the active feature from input
    pathway i to output pathway o.

    Parameters
    ----------
    N_pathways : int
        Number of input/output pathways (default 3).
    N_features : int
        Number of discrete features per pathway (default 3).
    samples_per_task : int
        Number of random stimuli to sample per task (default 100).
    sd_scale : float
        Scale for the diagonal per-pathway Gaussian noise (see Notes below).
    seed : int or None
        Seed for NumPy RNG (local to this function).

    Returns
    -------
    input_sgl : (5*samples_per_task, N_pathways*N_features) float32
        Noisy stimuli.
    tasks_sgl : (5*samples_per_task, N_pathways^2) float32
        One-hot task cues (exactly one active unit).
    train_sgl : (5*samples_per_task, N_pathways*N_features) float32
        One-hot targets derived deterministically from (stimulus, task).
    meta : dict
        {
          "task_names"        : ['A','B','C','D','E'],
          "structurally_dep"  : [('A','D'), ('A','E'), ('B','D'), ('B','E')],
          "functionally_dep"  : [('A','B')],
          "independent_pairs" : [('A','C'), ('B','C')],
          "input_masks"       : {name: binary mask before noise},
          "output_masks"      : {name: binary mask before noise},
          "task_indices"      : array of task names per row,
          "stimulus_indices"  : array of within-task stimulus indices,
        }

    Notes
    -----
    • Noise model matches `generate_task_patterns`: within-pathway diagonal
      Gaussian jitter whose variance row is selected by the active feature.
    • All tasks are equally represented (balanced) by construction.
    • This function is the one used for training in our pipeline.
    """

    if seed is not None:
        np.random.seed(seed)

    # dimensions
    I = N_pathways * N_features
    T = N_pathways ** 2
    R = I

    # define mapping
    task_map = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 2),
        'D': (0, 1),
        'E': (1, 0)
    }
    task_names = list(task_map.keys())

    # placeholders
    input_masks  = {}
    output_masks = {}
    all_inputs   = []
    all_tasks    = []
    all_outputs  = []
    task_indices = []
    stim_indices = []

    # build each task block
    for t_idx, name in enumerate(task_names):
        in_dim, out_dim = task_map[name]

        # one-hot task cue over T possible (in_dim * N_pathways + out_dim)
        cue = np.zeros(T)
        cue[in_dim * N_pathways + out_dim] = 1

        # generate structural binary masks
        X_bin = np.zeros((samples_per_task, I))
        Y_bin = np.zeros((samples_per_task, R))

        # random feature combinations to fill each pathway
        feats = np.random.randint(0, N_features, size=(samples_per_task, N_pathways))
        for i, fvec in enumerate(feats):
            for p in range(N_pathways):
                X_bin[i, p*N_features + fvec[p]] = 1
            # copy from input dim to output dim
            Y_bin[i, out_dim*N_features:(out_dim+1)*N_features] = \
                X_bin[i, in_dim*N_features:(in_dim+1)*N_features]

        # store the raw masks
        input_masks[name]  = X_bin.copy()
        output_masks[name] = Y_bin.copy()

        # tile cue & accumulate
        all_inputs.append(X_bin)
        all_tasks.append(np.tile(cue, (samples_per_task, 1)))
        all_outputs.append(Y_bin)

        task_indices.extend([name]*samples_per_task)
        stim_indices.extend(list(range(samples_per_task)))

    # stack into arrays
    input_sgl  = np.vstack(all_inputs)
    tasks_sgl  = np.vstack(all_tasks)
    train_sgl  = np.vstack(all_outputs)

    # add Gaussian noise to inputs
    sd_mat = np.random.rand(N_features, N_features) * sd_scale
    noisy_inputs = np.zeros_like(input_sgl)
    for i in range(len(input_sgl)):
        for p in range(N_pathways):
            block = input_sgl[i, p*N_features:(p+1)*N_features]
            feat = block.argmax()
            cov  = np.diag(sd_mat[feat])
            noisy_inputs[i, p*N_features:(p+1)*N_features] = \
                np.random.multivariate_normal(block, cov)

    # dependencies for easy selection
    structurally_dep = [('A','D'), ('A','E'), ('B','D'), ('B','E')]
    functionally_dep = [('A','B')]     # tasks that only depend via D/E
    independent_pairs = [('A','C'), ('B','C')]

    meta = {
        "task_names": task_names,
        "structurally_dep": structurally_dep,
        "functionally_dep": functionally_dep,
        "independent_pairs": independent_pairs,
        "input_masks": input_masks,
        "output_masks": output_masks,
        "task_indices": np.array(task_indices),
        "stimulus_indices": np.array(stim_indices),
    }

    return noisy_inputs, tasks_sgl, train_sgl, meta


import numpy as np

def _decode_task_from_vector(task_vec, N_pathways=3):
    M = task_vec.reshape(N_pathways, N_pathways) # removed .T here, check if correct
    i, o = np.argwhere(M == 1)[0]
    return int(i), int(o)

def _expected_target_from_mask(x_mask, i, o, N_features=3):
    """x_mask is the CLEAN one-hot stimulus per pathway (length N_pathways*N_features)."""
    y = np.zeros_like(x_mask)
    f = np.argmax(x_mask[i*N_features:(i+1)*N_features])
    y[o*N_features + f] = 1
    return y

def self_test_fixed(samples_per_task=1000, sd_scale=0.25, seed=0, N_pathways=3, N_features=3):
    """
    Mapping/balance checks that use CLEAN masks from meta (not argmax of the noisy inputs).
    """
    from prp.task_generator import generate_fixed_task_set
    noisy_inputs, tasks_sgl, train_sgl, meta = generate_fixed_task_set(
        N_pathways=N_pathways, N_features=N_features,
        samples_per_task=samples_per_task, sd_scale=sd_scale, seed=seed
    )

    # 1) Mapping correctness: for each row k, rebuild target from CLEAN mask
    task_names = list(meta["input_masks"].keys())
    name_for_k = meta["task_indices"]           # array of strings 'A'..'E', one per row
    stim_ix_for_k = meta["stimulus_indices"]    # within-task stimulus index, one per row

    mismatches = 0
    for k in range(len(noisy_inputs)):
        name = name_for_k[k]                    # e.g., 'A'
        sidx = int(stim_ix_for_k[k])            # which clean mask in that task
        x_mask = meta["input_masks"][name][sidx]   # CLEAN one-hot stimulus
        i, o = _decode_task_from_vector(tasks_sgl[k], N_pathways=N_pathways)
        y_expected = _expected_target_from_mask(x_mask, i, o, N_features=N_features)

        if not np.array_equal(y_expected, train_sgl[k]):
            mismatches += 1
            # Uncomment next two lines to inspect the first offending example:
            # print("First mismatch at row", k, "task", name, "stimidx", sidx, "i->o", (i,o))
            # break

    assert mismatches == 0, f"Mapping mismatch detected on {mismatches} samples!"
    print("✅ Mapping check passed (using clean masks).")

    # 2) Balance check
    from collections import Counter
    def key(tvec):
        return _decode_task_from_vector(tvec, N_pathways=N_pathways)
    counts = Counter([key(t) for t in tasks_sgl])
    print("✅ Balanced task counts:", counts)

def noise_flip_rate(samples_per_task=1000, sd_scale=0.25, seed=0, N_pathways=3, N_features=3):
    """
    How often does noise flip the argmax within a pathway? (Explains why the earlier test failed.)
    """
    from prp.task_generator import generate_fixed_task_set
    noisy_inputs, tasks_sgl, train_sgl, meta = generate_fixed_task_set(
        N_pathways=N_pathways, N_features=N_features,
        samples_per_task=samples_per_task, sd_scale=sd_scale, seed=seed
    )
    flips = 0
    total = 0
    for name, X_clean in meta["input_masks"].items():
        X_noisy = noisy_inputs[meta["task_indices"] == name]
        # align clean masks to the subset of rows for this task using stimulus_indices
        S = meta["stimulus_indices"][meta["task_indices"] == name]
        for row, sidx in enumerate(S):
            x_clean = X_clean[int(sidx)]
            x_noisy = X_noisy[row]
            for p in range(N_pathways):
                off = p * N_features
                f_clean = np.argmax(x_clean[off:off+N_features])
                f_noisy = np.argmax(x_noisy[off:off+N_features])
                flips += int(f_clean != f_noisy)
                total += 1
    rate = flips / max(total, 1)
    print(f"Noise flip rate (argmax differs from clean mask): {rate:.3%}")
