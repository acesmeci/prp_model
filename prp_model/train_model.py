from prp_model.task_generator import generate_fixed_task_set
from prp_model.nn_wrapper import TaskNetworkWrapper

inp, task, target, _ = generate_fixed_task_set(
    N_pathways=3,
    N_features=3,
    samples_per_task=100,
)

net = TaskNetworkWrapper(hidden_size=100)
net.train_online(inp, task, target, stop_loss=0.001)

