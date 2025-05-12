# PRP Model

A neural network model of the **Psychological Refractory Period (PRP)** effect, adapted from Musslick et al. (2023), implemented in PyTorch.

This project replicates Simulation 3 from the paper _"On the Rational Boundedness of Cognitive Control: Shared Versus Separated Representations"_ to explore how shared representations, graded conflict and persistence contribute to multi-task interference.

---

## Folder Structure

prp_model/
- lca.py # LCA decision dynamics
- nn_wrapper.py # Model training wrapper
- task_network.py # Feedforward network definition
- task_generator.py # Single-task pattern generation
- multitask_generator.py # Multitask (dual-task) pattern generation
- prp_simulator.py # Main PRP simulation + sweep_soa()
- training_utils.py # High-level training functions

---

## Features

- Continuous-time stimulus and task input stream
- Tau-modulated task control (`tau_net`, `tau_task`)
- LCA (Leaky Competing Accumulator) dynamics for decision-making
- Persistence integration to simulate temporal smoothing
- Single-task vs. multitask training toggle
- PRP curve evaluation (Task B RT and error vs. SOA)
- Hidden layer activation visualization (PCA & cosine similarity)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/acesmeci/prp_model.git
cd prp_model
```

### 2. Run inside a Python script or Jupyter/Colab notebook

```bash
from prp_model.prp_simulator import sweep_soa
from prp_model.training_utils import train_with_optional_multitask
```

### 3. Train and evaluate

Use `train_with_optional_multitask()` to train the model with or without multitasking.

Use `sweep_soa()` to simulate the PRP paradigm and analyze reaction times across SOAs.

## Reference
Musslick, S., et al. (2023).
On the Rational Boundedness of Cognitive Control: Shared Versus Separated Representations.
