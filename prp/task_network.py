# task_network.py
"""Core task-modulated feedforward network (Sim Study 3).

Implements the 3-layer architecture from Musslick et al. (2023), Fig. 9:
    stim_input → hidden → output
                ↑          ↑
             task_input  task_input

Key details:
- Both hidden and output layers receive additive task-control input.
- Fixed bias offset of -2.0 added to net inputs at hidden and output.
- Sigmoid activations; all weights init ~ U[-0.5, 0.5].
- Task cues are encoded in ROW-MAJOR order: index = in_dim * N_pathways + out_dim.
- Designed to be wrapped by `TaskNetworkWrapper` for online training and
  time-course integration (with persistence) used in PRP simulations.
"""


import torch
import torch.nn as nn

class TaskNetwork(nn.Module):
    """
    3-layer feedforward network with task-control inputs.
    Architecture (Fig.9, Musslick et al. 2023):
      stim_input (one-hot) ──► hidden ──► output ──► responses
                         ▲                ▲
           task_input ──┘                └──► output
    Both hidden and output layers include a fixed bias offset of -2.
    All weights initialized uniformly in [-0.5, 0.5].
    """

    def __init__(
        self,
        stim_input_dim: int,
        task_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias_offset: float = -2.0,
    ):
        """
        :param stim_input_dim: dimensionality of stimulus input (e.g., one-hot size)
        :param task_input_dim: dimensionality of task input (one-hot for task cues)
        :param hidden_dim: number of hidden units (e.g., 100)
        :param output_dim: number of output units (one-hot for responses)
        :param bias_offset: fixed offset added to net inputs (default -2.0)
        """
        super().__init__()
        self.bias_offset = bias_offset

        # Stimulus-to-hidden and Task-to-hidden (no learnable biases here)
        self.fc_input_hidden = nn.Linear(stim_input_dim, hidden_dim, bias=False)
        self.fc_task_hidden  = nn.Linear(task_input_dim, hidden_dim, bias=False)

        # Hidden-to-output and Task-to-output
        self.fc_hidden_output = nn.Linear(hidden_dim, output_dim, bias=False)
        self.fc_task_output   = nn.Linear(task_input_dim, output_dim, bias=False)

        # Sigmoid activation
        self.act = nn.Sigmoid()

        # Initialize all weights uniformly in [-0.5, 0.5]
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.5, 0.5)

    def forward(self, stim_input: torch.Tensor, task_input: torch.Tensor):
        """
        :param stim_input: tensor of shape (batch, stim_input_dim)
        :param task_input: tensor of shape (batch, task_input_dim)
        :returns:
          y_o: output activations (batch, output_dim)
          y_h: hidden  activations (batch, hidden_dim)  [optional for later LCA]
        """
        # --- Hidden layer net input & activation ---
        net_h = self.fc_input_hidden(stim_input) \
              + self.fc_task_hidden(task_input) \
              + self.bias_offset
        y_h   = self.act(net_h)

        # --- Output layer net input & activation ---
        net_o = self.fc_hidden_output(y_h) \
              + self.fc_task_output(task_input) \
              + self.bias_offset
        y_o   = self.act(net_o)

        return y_o, y_h
