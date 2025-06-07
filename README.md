# PRP Model

A neural network model of the **Psychological Refractory Period (PRP)** effect, adapted from Musslick et al. (2023), implemented in PyTorch.

This project replicates **Simulation Study 3** from:

> Musslick, S., et al. (2023).  
> *On the Rational Boundedness of Cognitive Control: Shared Versus Separated Representations.*

The model investigates how shared task representations, conflict, and control policies affect multitasking performance under dual-task interference.

---

## 📁 Project Structure

PRP_Model/
├── prp/ # Core model modules
│ ├── task_network.py # Feedforward network architecture
│ ├── nn_wrapper.py # Training + integration interface
│ ├── lca.py # LCA decision dynamics
│ ├── task_generator.py # Single-task pattern generation (A–E)
│ ├── multitask_generator.py # Valid dual-task combinations
│ ├── prp_simulator.py # PRP trial logic and sweep_soa
│ ├── choose_onset_policy.py # Reward-rate-optimized Task 2 onset
│ └── training_utils.py # High-level training wrappers
│
├── scripts/ # CLI entry points
│ └── train_model.py # Training pipeline with early stopping
│
├── notebooks/ # Experimental notebooks
├── output/ # Plots, saved models, etc.
├── requirements.txt
└── README.md

---

## 🚀 Features

- ✅ Feedforward NN with task-based modulation (`tau_net`, `tau_task`)
- ✅ **Online training** with MSE loss and fixed bias terms
- ✅ Decision dynamics via **Leaky Competing Accumulator (LCA)**
- ✅ **Persistence** modeling for temporal integration
- ✅ Optional multitask pretraining
- ✅ PRP behavior simulation via `sweep_soa()`
- ✅ **Reward-maximizing Task 2 onset policy**
- ✅ Diagnostic tools for:
  - Single-task decoding
  - Multitask interference
  - Hidden layer structure (correlation matrix)

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/acesmeci/prp_model.git
cd prp_model
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model

```bash
PYTHONPATH=$(pwd) python scripts/train_model.py
```

### 4. Use in a notebook

```bash
from prp.nn_wrapper import TaskNetworkWrapper
net = TaskNetworkWrapper()
net.model.load_state_dict(torch.load("output/trained_model.pth"))

from prp.task_generator import generate_fixed_task_set
from prp.prp_simulator import sweep_soa
```

## Reference
Musslick, S., et al. (2023).
On the Rational Boundedness of Cognitive Control: Shared Versus Separated Representations.
