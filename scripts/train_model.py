from prp.task_generator import generate_fixed_task_set
from prp.nn_wrapper import TaskNetworkWrapper
import torch

inp, task, target, _ = generate_fixed_task_set(
    N_pathways=3,
    N_features=3,
    samples_per_task=100,
)

net = TaskNetworkWrapper(hidden_size=100)
net.train_online(inp, task, target, stop_loss=0.005) # Paper stop_loss = 0.001

# Save model after training
torch.save(net.model.state_dict(), "output/trained_model.pth")
print("✅ Model saved to output/trained_model.pth")


