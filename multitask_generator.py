import numpy as np
from itertools import combinations

def generate_multitask_patterns(N_pathways, N_features, samples_per_task, relevant_tasks):
    I = N_pathways * N_features
    T = N_pathways ** 2
    R = I

    # All 2-task combinations (no overlapping inputs or outputs)
    valid_combs = []
    for comb in combinations(relevant_tasks, 2):
        task_vec_1 = np.zeros(T)
        task_vec_1[comb[0] - 1] = 1
        M1 = task_vec_1.reshape(N_pathways, N_pathways).T
        in1, out1 = np.argwhere(M1 == 1)[0]

        task_vec_2 = np.zeros(T)
        task_vec_2[comb[1] - 1] = 1
        M2 = task_vec_2.reshape(N_pathways, N_pathways).T
        in2, out2 = np.argwhere(M2 == 1)[0]

        # Accept only non-overlapping tasks
        if in1 != in2 and out1 != out2:
            valid_combs.append(comb)

    input_multi = []
    tasks_multi = []
    train_multi = []
    task_comb_ids = []

    for comb_id, (t1, t2) in enumerate(valid_combs):
        task_vec = np.zeros(T)
        task_vec[t1 - 1] = 1
        task_vec[t2 - 1] = 1

        for _ in range(samples_per_task):
            stim = np.zeros(I)
            train = np.zeros(R)

            # Stimulus: one feature per input dim
            features = np.random.randint(0, N_features, size=N_pathways)
            for i in range(N_pathways):
                stim[i * N_features + features[i]] = 1

            # Outputs for t1
            tm1 = np.zeros(T)
            tm1[t1 - 1] = 1
            in1, out1 = np.argwhere(tm1.reshape(N_pathways, N_pathways).T == 1)[0]
            train[out1 * N_features + features[in1]] = 1

            # Outputs for t2
            tm2 = np.zeros(T)
            tm2[t2 - 1] = 1
            in2, out2 = np.argwhere(tm2.reshape(N_pathways, N_pathways).T == 1)[0]
            train[out2 * N_features + features[in2]] = 1

            input_multi.append(stim)
            tasks_multi.append(task_vec)
            train_multi.append(train)
            task_comb_ids.append(comb_id)

    return (
        np.array(input_multi),
        np.array(tasks_multi),
        np.array(train_multi),
        {
            "taskCombs": valid_combs,
            "taskIdx": np.array(task_comb_ids)
        }
    )
