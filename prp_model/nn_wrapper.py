import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from prp_model.task_network import TaskNetwork

class TaskNetworkWrapper:
    def __init__(self,
                 input_size=18,
                 hidden_size=100,
                 output_size=9,
                 learning_rate=0.001,
                 activation="sigmoid", # I changed this to sigmoid June 1st, 2025
                 device="cpu"):

        self.device = torch.device(device)
        self.model = TaskNetwork(input_size, hidden_size, output_size, activation).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Logging
        self.loss_log = []
        self.accuracy_log = []

    def train_online(self, stim_inputs, task_inputs, targets, n_epochs=100):
        """
        Performs online training on a set of patterns.

        Args:
            stim_inputs: NumPy array of shape (N, 9)
            task_inputs: NumPy array of shape (N, 9)
            targets:     NumPy array of shape (N, 9)
        """
        stim_inputs = torch.tensor(stim_inputs, dtype=torch.float32).to(self.device)
        task_inputs = torch.tensor(task_inputs, dtype=torch.float32).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

        dataset_size = stim_inputs.shape[0]

        for epoch in range(n_epochs):
            total_loss = 0
            correct = 0

            for i in range(dataset_size):
                s = stim_inputs[i].unsqueeze(0)
                t = task_inputs[i].unsqueeze(0)
                y_true = targets[i].unsqueeze(0)

                self.optimizer.zero_grad()
                y_pred = self.model(s, t)

                loss = self.loss_fn(y_pred, y_true)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Check if argmax matches
                if torch.argmax(y_pred) == torch.argmax(y_true):
                    correct += 1

            avg_loss = total_loss / dataset_size
            acc = correct / dataset_size

            self.loss_log.append(avg_loss)
            self.accuracy_log.append(acc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}")

    def predict(self, stim_input, task_input):
        """
        Returns the raw output of the model for a single pattern.
        """
        self.model.eval()
        with torch.no_grad():
            s = torch.tensor(stim_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            t = torch.tensor(task_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(s, t)
        return output.cpu().numpy().flatten()

    def integrate(self, input_series, task_series,
              tau_net=1.0, tau_task=1.0, persistence=0.0, 
              return_tensor=False):
        """
        Simulates network forward pass over a time series with optional persistence.

        Args:
            input_series: list of input vectors (length T)
            task_series: list of task vectors (length T)
            tau_net: task-to-hidden strength
            tau_task: task-to-output strength
            persistence: rate of net input integration (0 = no persistence)

        Returns:
            outputs: list of output activations over time
        """
        self.model.eval()
        outputs = []

        hidden_prev_net = None
        output_prev_net = None

        with torch.no_grad():
            for s_t, t_t in zip(input_series, task_series):
                stim = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.device)
                task = torch.tensor(t_t, dtype=torch.float32).unsqueeze(0).to(self.device)

                # --- Hidden Layer ---
                net_hidden = self.model.fc_input_to_hidden(torch.cat([stim, task], dim=1)) + \
                             tau_net * self.model.task_to_hidden(task) - 2.0 # Added -2 bias term


                if hidden_prev_net is not None:
                    net_hidden = (1 - persistence) * net_hidden + persistence * hidden_prev_net

                hidden = self.model.activation_fn(net_hidden)
                hidden_prev_net = net_hidden  # Save for next time step

                # --- Output Layer ---
                net_output = self.model.fc_hidden_to_output(hidden) + tau_task * self.model.task_to_output(task) - 2.0 # Added -2 bias term

                if output_prev_net is not None:
                    net_output = (1 - persistence) * net_output + persistence * output_prev_net

                output = net_output
                output_prev_net = net_output

                if return_tensor:
                    output.requires_grad_()
                    outputs.append(output)
                else:
                    outputs.append(output.detach().cpu().numpy().flatten())

        return outputs



    def get_weights(self):
        """
        Returns model weights for inspection or analysis.
        """
        return {
            "input_to_hidden": self.model.fc_input_to_hidden.weight.data.cpu().numpy(),
            "hidden_to_output": self.model.fc_hidden_to_output.weight.data.cpu().numpy()
        }
