import torch
import torch.nn as nn
import torch.optim
from prp.task_network import TaskNetwork

class TaskNetworkWrapper:
    def __init__(self,
                 input_size=18,
                 hidden_size=100,
                 output_size=9,
                 learning_rate=0.3,
                 activation="sigmoid",
                 device="cpu"):

        self.device = torch.device(device)
        self.model = TaskNetwork(input_size, hidden_size, output_size, activation).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # === Fixed Bias Initialization ===
        with torch.no_grad():
            self.model.fc_input_to_hidden.bias.fill_(-2.0)
            self.model.fc_hidden_to_output.bias.fill_(-2.0)
            if hasattr(self.model, "task_to_hidden") and self.model.task_to_hidden.bias is not None:
                self.model.task_to_hidden.bias.fill_(0.0)
            if hasattr(self.model, "task_to_output") and self.model.task_to_output.bias is not None:
                self.model.task_to_output.bias.fill_(0.0)

        # === Freeze Biases ===
        self.model.fc_input_to_hidden.bias.requires_grad = False
        self.model.fc_hidden_to_output.bias.requires_grad = False

        # Logging
        self.loss_log = []
        self.accuracy_log = []

    def train_online(self, stim_inputs, task_inputs, targets,
                 max_epochs=5000, stop_loss=0.001, print_every=10):
        """
        Trains the network until MSE drops below `stop_loss`, or until `max_epochs`.

        Args:
            stim_inputs: (N, 9)
            task_inputs: (N, 9)
            targets:     (N, 9)
        """
        stim_inputs = torch.tensor(stim_inputs, dtype=torch.float32).to(self.device)
        task_inputs = torch.tensor(task_inputs, dtype=torch.float32).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

        dataset_size = stim_inputs.shape[0]
        epoch = 0

        while epoch < max_epochs:
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
                if torch.argmax(y_pred) == torch.argmax(y_true):
                    correct += 1

            avg_loss = total_loss / dataset_size
            acc = correct / dataset_size

            self.loss_log.append(avg_loss)
            self.accuracy_log.append(acc)

            if epoch % print_every == 0:
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}")

            if avg_loss <= stop_loss:
                print(f"✅ Early stopping at epoch {epoch} | Final Loss: {avg_loss:.4f}")
                break

            epoch += 1

        if epoch == max_epochs:
            print(f"⚠️ Max epochs reached ({max_epochs}) | Final Loss: {avg_loss:.4f}")


    def predict(self, stim_input, task_input):
        self.model.eval()
        with torch.no_grad():
            s = torch.tensor(stim_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            t = torch.tensor(task_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(s, t)
        return output.cpu().numpy().flatten()

    def integrate(self, input_series, task_series,
                  tau_net=0.2, tau_task=0.2, persistence=0.0, 
                  return_tensor=False):
        self.model.eval()
        outputs = []
        hidden_prev_net = None
        output_prev_net = None

        with torch.no_grad():
            for s_t, t_t in zip(input_series, task_series):
                stim = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.device)
                task = torch.tensor(t_t, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Hidden layer input
                net_hidden = self.model.fc_input_to_hidden(torch.cat([stim, task], dim=1)) + \
                             tau_net * self.model.task_to_hidden(task)

                if hidden_prev_net is not None:
                    net_hidden = (1 - persistence) * net_hidden + persistence * hidden_prev_net
                hidden = self.model.activation_fn(net_hidden)
                hidden_prev_net = net_hidden

                # Output layer input
                net_output = self.model.fc_hidden_to_output(hidden) + \
                             tau_task * self.model.task_to_output(task)

                if output_prev_net is not None:
                    net_output = (1 - persistence) * net_output + persistence * output_prev_net

                if return_tensor:
                    net_output.requires_grad_()
                    outputs.append(net_output)
                else:
                    outputs.append(net_output.cpu().numpy().flatten())
                output_prev_net = net_output

        return outputs

    def get_weights(self):
        return {
            "input_to_hidden": self.model.fc_input_to_hidden.weight.data.cpu().numpy(),
            "hidden_to_output": self.model.fc_hidden_to_output.weight.data.cpu().numpy()
        }

    def get_hidden_activation(self, stim_input, task_input, tau_net=1.0):
        x = torch.cat((stim_input, task_input), dim=0).unsqueeze(0).to(self.device)
        t = task_input.unsqueeze(0).to(self.device)
        net_hidden = self.model.fc_input_to_hidden(x) + tau_net * self.model.task_to_hidden(t)
        h = self.model.activation_fn(net_hidden)
        return h.squeeze().detach().cpu().numpy()

    def get_output_net_input_series(self, input_series, task_series,
                                    tau_net=1.0, tau_task=1.0, persistence=0.0):
        self.model.eval()
        net_outputs = []
        hidden_prev_net = None
        output_prev_net = None

        with torch.no_grad():
            for s_t, t_t in zip(input_series, task_series):
                stim = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.device)
                task = torch.tensor(t_t, dtype=torch.float32).unsqueeze(0).to(self.device)

                net_hidden = self.model.fc_input_to_hidden(torch.cat([stim, task], dim=1)) + \
                             tau_net * self.model.task_to_hidden(task)

                if hidden_prev_net is not None:
                    net_hidden = (1 - persistence) * net_hidden + persistence * hidden_prev_net
                hidden = self.model.activation_fn(net_hidden)
                hidden_prev_net = net_hidden

                net_output = self.model.fc_hidden_to_output(hidden) + \
                             tau_task * self.model.task_to_output(task)

                if output_prev_net is not None:
                    net_output = (1 - persistence) * net_output + persistence * output_prev_net

                net_outputs.append(net_output.squeeze().cpu().numpy())
                output_prev_net = net_output

        return net_outputs
