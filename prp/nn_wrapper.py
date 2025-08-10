# nn_wrapper.py

import torch
import torch.nn as nn
from prp.task_network import TaskNetwork


class TaskNetworkWrapper:
    """
    Wraps the TaskNetwork to handle training, prediction, and
    time-course integration (for PRP simulations).
    """

    def __init__(
        self,
        stim_input_dim: int,
        task_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.3,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = TaskNetwork(
            stim_input_dim=stim_input_dim,
            task_input_dim=task_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.loss_log = []
        self.accuracy_log = []

    def train_online(
        self,
        stim_inputs: torch.Tensor,
        task_inputs: torch.Tensor,
        targets: torch.Tensor,
        max_epochs: int = 8000,
        stop_loss: float = 1e-3,
        print_every: int = 10,
    ):
        """
        Trains on (stim, task) -> one-hot target pairs until MSE loss <= stop_loss
        or max_epochs is reached.

        stim_inputs: (N, stim_input_dim) float tensor
        task_inputs: (N, task_input_dim) float tensor
        targets:     (N, output_dim)     float tensor (one-hot)
        """
        self.model.train()
        stim = stim_inputs.to(self.device)
        task = task_inputs.to(self.device)
        y_true = targets.to(self.device)

        for epoch in range(max_epochs):
            total_loss = 0.0
            correct = 0

            # Shuffle each epoch to avoid order effects
            perm = torch.randperm(stim.size(0))
            for i in perm:
                x_i = stim[i : i + 1]
                t_i = task[i : i + 1]
                y_i = y_true[i : i + 1]

                self.optimizer.zero_grad()
                y_pred, _ = self.model(x_i, t_i)
                loss = self.loss_fn(y_pred, y_i)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct += (y_pred.argmax(dim=1) == y_i.argmax(dim=1)).sum().item()

            avg_loss = total_loss / stim.size(0)
            acc = correct / stim.size(0)
            self.loss_log.append(avg_loss)
            self.accuracy_log.append(acc)

            if epoch % print_every == 0:
                print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")

            if avg_loss <= stop_loss:
                print(f"✅ Converged at epoch {epoch:04d} | Loss: {avg_loss:.4f}")
                break

        else:
            print(f"⚠️ Max epochs ({max_epochs}) reached | Final loss: {avg_loss:.4f}")

    def predict(self, stim_input: torch.Tensor, task_input: torch.Tensor):
        """
        Single-timestep prediction: returns output vector.
        stim_input: (stim_input_dim,)
        task_input: (task_input_dim,)
        """
        self.model.eval()
        with torch.no_grad():
            x = stim_input.unsqueeze(0).to(self.device)
            t = task_input.unsqueeze(0).to(self.device)
            y_pred, _ = self.model(x, t)
        return y_pred.squeeze(0).cpu()

    def integrate(
        self,
        stim_sequence: torch.Tensor,
        task_sequence: torch.Tensor,
        persistence: float = 0.0,
    ):
        """
        Runs the network over a sequence of timesteps, applying
        temporal persistence to net inputs. Returns a list of
        output activations for each timestep.

        stim_sequence: (T, stim_input_dim)
        task_sequence: (T, task_input_dim)
        persistence:   a float in [0,1] controlling leak (p in paper)
        """
        self.model.eval()
        outputs = []

        # placeholders for previous net inputs
        prev_net_h = None
        prev_net_o = None

        with torch.no_grad():
            for t_idx in range(stim_sequence.size(0)):
                x_t = stim_sequence[t_idx : t_idx + 1].to(self.device)
                t_t = task_sequence[t_idx : t_idx + 1].to(self.device)

                # compute raw net_h
                net_h = (
                    self.model.fc_input_hidden(x_t)
                    + self.model.fc_task_hidden(t_t)
                    + self.model.bias_offset
                )
                if prev_net_h is not None:
                    net_h = (1 - persistence) * net_h + persistence * prev_net_h
                y_h = torch.sigmoid(net_h)
                prev_net_h = net_h

                # compute raw net_o
                net_o = (
                    self.model.fc_hidden_output(y_h)
                    + self.model.fc_task_output(t_t)
                    + self.model.bias_offset
                )
                if prev_net_o is not None:
                    net_o = (1 - persistence) * net_o + persistence * prev_net_o
                y_o = torch.sigmoid(net_o)
                prev_net_o = net_o

                outputs.append(y_o.squeeze(0).cpu())

        return outputs

    # --- Optional debugging methods ---

    def get_weights(self):
        return {
            "W_input_hidden": self.model.fc_input_hidden.weight.detach().cpu().numpy(),
            "W_task_hidden":  self.model.fc_task_hidden.weight.detach().cpu().numpy(),
            "W_hidden_output":self.model.fc_hidden_output.weight.detach().cpu().numpy(),
            "W_task_output":  self.model.fc_task_output.weight.detach().cpu().numpy(),
        }

    def logs(self):
        """Returns (loss_log, accuracy_log)."""
        return self.loss_log, self.accuracy_log
