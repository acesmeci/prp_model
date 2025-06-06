import torch

def initialize_shared_task_to_hidden_weights(model, stimulus_task_map, hidden_size=100, share_ratio=0.5, weight_value=1.0, seed=42):
    """
    Initialize task-to-hidden weights to induce fixed representational sharing.

    Args:
        model: TaskNetwork instance (not the wrapper).
        stimulus_task_map: dict mapping stimulus dimension (0, 1, 2) to list of task indices (0–8).
                           Tasks that rely on the same stimulus dimension (e.g. A and D) will share hidden units.
        hidden_size: number of hidden units in the model.
        share_ratio: fraction of hidden units that will be shared per group.
        weight_value: fixed value of weights to assign for shared units (typically 1.0).
        seed: for reproducibility.

    Example:
        stimulus_task_map = {
            0: [0, 3],  # Task A and D → S0
            1: [4, 1],  # Task B and E → S1
            2: [8]      # Task C → S2 (independent)
        }
    """
    torch.manual_seed(seed)
    W = torch.zeros((hidden_size, 9))  # shape: [hidden_size, num_task_units]

    units_per_group = int(hidden_size * share_ratio)
    current_offset = 0

    for stim_dim, task_indices in stimulus_task_map.items():
        if current_offset + units_per_group > hidden_size:
            raise ValueError("Too many shared groups for given hidden size and share_ratio.")
        
        for task_idx in task_indices:
            W[current_offset:current_offset + units_per_group, task_idx] = weight_value

        current_offset += units_per_group

    # Assign remaining hidden units randomly
    for task_idx in range(9):
        if torch.count_nonzero(W[:, task_idx]) < hidden_size:
            unassigned = (W[:, task_idx] == 0).nonzero(as_tuple=True)[0]
            W[unassigned, task_idx] = torch.randn(len(unassigned)) * 0.1

    with torch.no_grad():
        model.task_to_hidden.weight.copy_(W)

    return W  # for inspection
