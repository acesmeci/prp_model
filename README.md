# PRP Model

A neural network model of the **Psychological Refractory Period (PRP)** effect, adapted from Musslick et al. (2023), implemented in PyTorch.

This project replicates key simulations from the paper _"On the Rational Boundedness of Cognitive Control: Shared Versus Separated Representations"_ to explore how shared representations and limited control contribute to dual-task interference.

---

## 🚧 Features

- Continuous-time stimulus and task input stream
- Tau-modulated task control (`tau_net`, `tau_task`)
- LCA (Leaky Competing Accumulator) dynamics for decision-making
- Persistence integration to simulate temporal smoothing
- Single-task vs. multitask training toggle
- PRP curve evaluation (Task B RT and error vs. SOA)
- Hidden layer activation visualization (PCA & cosine similarity)

---

## 📁 Folder Structure

prp_model/
├── lca.py # LCA decision dynamics
├── nn_wrapper.py # Model training wrapper
├── task_network.py # Feedforward network definition
├── task_generator.py # Single-task pattern generation
├── multitask_generator.py # Multitask (dual-task) pattern generation
├── prp_simulator.py # Main PRP simulation + sweep_soa()
├── training_utils.py # High-level training functions
├── init.py

---

## 🧠 Key Concepts

- **PRP Effect**: Delayed RT for Task B as SOA decreases due to bottlenecked control
- **Tau Modulation**: Scales the influence of task control signals
- **Persistence**: Smooths net input over time; critical for delayed interference
- **Functional Dependence**: Interference only emerges when tasks share representations

---

## 🚀 Quick Start

git clone https://github.com/your-username/prp_model.git
cd prp_model

In your notebook or script:

from prp_model.prp_simulator import sweep_soa
from prp_model.training_utils import train_with_optional_multitask

## Reference
Musslick, S., et al. (2023).
On the Rational Boundedness of Cognitive Control: Shared Versus Separated Representations.
