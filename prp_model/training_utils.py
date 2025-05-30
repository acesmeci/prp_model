import numpy as np
from prp_model.task_generator import generate_task_patterns
from prp_model.multitask_generator import generate_multitask_patterns

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


# Train with optional tau control and persistence settings
def train_with_control_config(task_net, train_data_fn, n_epochs=100,
                              tau_net=0.0, tau_task=0.0, persistence=0.0,
                              verbose=True):
    """
    Train the task network under specific control settings to encourage representational overlap.

    Args:
        task_net: TaskNetworkWrapper instance
        train_data_fn: function returning (stim_inputs, task_inputs, targets)
        n_epochs: number of epochs
        tau_net: task-to-hidden influence
        tau_task: task-to-output influence
        persistence: carryover rate in hidden/output
        verbose: print training logs every 10 epochs
    """
    import torch

    stim_inputs, task_inputs, targets = train_data_fn()
    stim_inputs = torch.tensor(stim_inputs, dtype=torch.float32).to(task_net.device)
    task_inputs = torch.tensor(task_inputs, dtype=torch.float32).to(task_net.device)
    targets = torch.tensor(targets, dtype=torch.float32).to(task_net.device)

    task_net.model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0

        for i in range(stim_inputs.shape[0]):
            input_series = [stim_inputs[i].cpu().numpy()] * 2
            task_series = [task_inputs[i].cpu().numpy()] * 2
            target = targets[i].unsqueeze(0)

            output_series = task_net.integrate(input_series, task_series,
                                   tau_net=tau_net, tau_task=tau_task,
                                   persistence=persistence, return_tensor=True)
            output = output_series[-1]


            task_net.optimizer.zero_grad()
            loss = task_net.loss_fn(output, target)
            loss.backward()
            task_net.optimizer.step()

            total_loss += loss.item()
            correct += int(torch.argmax(output) == torch.argmax(target))

        avg_loss = total_loss / stim_inputs.shape[0]
        acc = correct / stim_inputs.shape[0]
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}")
