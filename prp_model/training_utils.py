import numpy as np

def train_with_optional_multitask(net,
                                   N_pathways=3,
                                   N_features=3,
                                   samples_per_task=5,
                                   relevant_tasks=[1, 2, 4, 5, 9],
                                   pretrain_multitask=True,
                                   n_epochs=200):
    """
    Train the TaskNetworkWrapper with optional multitask pretraining.

    Args:
        net: TaskNetworkWrapper instance
        pretrain_multitask: If True, include multitask data in training
        All other args define the task parameters
    """

    from task_generator import generate_task_patterns
    from multitask_generator import generate_multitask_patterns

    # Single-task data
    input_sgl, tasks_sgl, train_sgl, _ = generate_task_patterns(
        N_pathways=N_pathways,
        N_features=N_features,
        samples_per_task=samples_per_task,
        relevant_tasks=relevant_tasks
    )

    if pretrain_multitask:
        # Multitask data
        input_multi, tasks_multi, train_multi, _ = generate_multitask_patterns(
            N_pathways=N_pathways,
            N_features=N_features,
            samples_per_task=samples_per_task,
            relevant_tasks=relevant_tasks
        )

        # Combine both
        input_train = np.vstack([input_sgl, input_multi])
        tasks_train = np.vstack([tasks_sgl, tasks_multi])
        train_output = np.vstack([train_sgl, train_multi])
    else:
        # Only single-task training
        input_train = input_sgl
        tasks_train = tasks_sgl
        train_output = train_sgl

    # Train the model
    net.train_online(input_train, tasks_train, train_output, n_epochs=n_epochs)
