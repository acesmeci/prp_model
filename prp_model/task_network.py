import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskNetwork(nn.Module):
    def __init__(self, input_size=18, hidden_size=100, output_size=9, activation="sigmoid"):
        super(TaskNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc_input_to_hidden = nn.Linear(input_size, hidden_size)
        self.fc_hidden_to_output = nn.Linear(hidden_size, output_size)

        self.task_to_hidden = nn.Linear(9, hidden_size, bias=False)
        self.task_to_output = nn.Linear(9, output_size, bias=False)

        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "sigmoid":
            self.activation_fn = torch.sigmoid
        else:
            raise ValueError("Unsupported activation: choose 'relu' or 'sigmoid'")

    def forward(self, stim_input, task_input, tau_net=1.0, tau_task=1.0):
        """
        stim_input: (batch_size, 9)
        task_input: (batch_size, 9)
        tau_net: scaling of task→hidden
        tau_task: scaling of task→output
        """
        x = torch.cat((stim_input, task_input), dim=1)
        
        net_hidden = self.fc_input_to_hidden(x) + tau_net * self.task_to_hidden(task_input) - 2.0
        h = self.activation_fn(net_hidden)

        net_output = self.fc_hidden_to_output(h) + tau_task * self.task_to_output(task_input) - 2.0
        return net_output


