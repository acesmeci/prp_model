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
    Generate training data for single-task and multitask PRP trials.

    Returns:
        - input_sgl: Noisy stimuli (single task)
        - tasks_sgl: One-hot task vectors
        - train_sgl: Correct outputs
        - meta: dict with clean versions, task/stim indices, and multitask set
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
        curr_tasksM = task_vector.reshape(N_pathways, N_pathways).T
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
    Generate a fixed set of 5 tasks (A–E) with structural/functional dependencies.

    Task definitions (Fig.13, Musslick et al.):
      A: S0 -> R0
      B: S1 -> R1
      C: S2 -> R2
      D: S0 -> R1 (shares S with A, R with B)
      E: S1 -> R0 (shares S with B, R with A)

    Returns:
      input_sgl      : (5*samples_per_task, N_pathways*N_features) noisy inputs
      tasks_sgl      : (5*samples_per_task, T) one-hot task cues (T = N_pathways^2)
      train_sgl      : (5*samples_per_task, N_pathways*N_features) one-hot correct outputs
      meta           : {
        "task_names"           : ['A','B','C','D','E'],
        "structurally_dep"     : [('A','D'), ('A','E'), ('B','D'), ('B','E')],
        "functionally_dep"     : [('A','B')],   # via D/E
        "independent_pairs"    : [('A','C'), ('B','C')],
        "input_masks"          : {...},        # raw binary masks before noise
        "output_masks"         : {...}, 
        "task_indices"         : [...],        # length = 5*samples_per_task
        "stimulus_indices"     : [...]
      }
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
