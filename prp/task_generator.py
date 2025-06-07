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

def generate_fixed_task_set(
    N_pathways=3,
    N_features=3,
    samples_per_task=100, # Paper (Sim 2 & 3) used 100 samples per task
    sd_scale=0.25,
    seed=None
):
    """
    Generate a fixed set of tasks (A–E) with known structural/functional dependencies.

    Task mapping:
        - Task A: S0 -> R0 (structurally independent)
        - Task B: S1 -> R1 (structurally independent)
        - Task C: S2 -> R2 (structurally independent)
        - Task D: S0 -> R1 (shares stimulus with A, output with B)
        - Task E: S1 -> R0 (shares stimulus with B, output with A)
    
    Returns:
        - input_sgl: noisy stimuli
        - tasks_sgl: one-hot task vectors
        - train_sgl: correct output vectors
        - meta: dictionary with input/output masks and task/stim indices
    """
    if seed is not None:
        np.random.seed(seed)

    I = N_pathways * N_features
    T = N_pathways ** 2
    R = I  # output size = input size (3x3)

    task_map = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 2),
        'D': (0, 1),
        'E': (1, 0)
    }

    task_ids = list(task_map.keys())
    stim_mask_list = []
    task_mask_list = []
    train_mask_list = []
    task_idx_list = []
    stim_idx_list = []

    for idx, task_name in enumerate(task_ids):
        in_dim, out_dim = task_map[task_name]
        task_vector = np.zeros(T)
        task_vector[in_dim * N_pathways + out_dim] = 1

        feature_combs = np.random.randint(0, N_features, size=(samples_per_task, N_pathways))
        curr_input_mask = np.zeros((samples_per_task, I))
        curr_output_mask = np.zeros((samples_per_task, R))

        for i, feat_idx in enumerate(feature_combs):
            for d in range(N_pathways):
                curr_input_mask[i, d * N_features + feat_idx[d]] = 1

            # Copy relevant input dim to output dim (task-defined mapping)
            in_start = in_dim * N_features
            out_start = out_dim * N_features
            curr_output_mask[i, out_start:out_start + N_features] = curr_input_mask[i, in_start:in_start + N_features]

        stim_mask_list.append(curr_input_mask)
        task_mask_list.append(np.tile(task_vector, (samples_per_task, 1)))
        train_mask_list.append(curr_output_mask)
        task_idx_list.append(np.full((samples_per_task,), task_name))
        stim_idx_list.append(np.arange(samples_per_task))

    input_sgl_mask = np.vstack(stim_mask_list)
    tasks_sgl_mask = np.vstack(task_mask_list)
    train_sgl_mask = np.vstack(train_mask_list)
    tasks_idx_sgl = np.concatenate(task_idx_list)
    stim_idx_sgl = np.concatenate(stim_idx_list)

    # Add noise to stimuli
    input_sgl = np.zeros_like(input_sgl_mask)
    sd_matrix = np.random.rand(N_features, N_features) * sd_scale

    for row_idx in range(input_sgl.shape[0]):
        for dim in range(N_pathways):
            col_start = dim * N_features
            col_end = (dim + 1) * N_features
            mu = input_sgl_mask[row_idx, col_start:col_end]
            active_feat = np.argmax(mu)
            input_sgl[row_idx, col_start:col_end] = np.random.multivariate_normal(mu, np.diag(sd_matrix[active_feat]))

    return input_sgl, tasks_sgl_mask, train_sgl_mask, {
        "input_sgl_mask": input_sgl_mask,
        "tasks_idx": tasks_idx_sgl,
        "stim_idx": stim_idx_sgl
    }
