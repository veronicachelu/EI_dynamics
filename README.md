# The Role of E/I modulation in Shaping Neural Dynamics in Decision-Making and Adaptive Learning

## Overview

This repository contains an analysis of the role of excitation and inhibition in shaping neural dynamics in two settings:
* **Decision-Making*** --- using a mean-field Wilson-Cowan rate model
* Adaptive Learning in Dynamic Environments (Continual Reinforcement Learning in a changing sequence of Probabilistic Reversal Learning (PRL) tasks) --- using a Neural Actor Critic model with an E/I modulation mechanism 

## Directory Structure

### 1. `mean_field__decision_making/`
This folder contains notebooks and scripts for simulating and analyzing neural dynamics in decision-making processes using mean-field approximations.

#### Files:
- **`evidence_accum_perf_crit.ipynb`**: For understanding evidence accumulation in achieving performance criteria during decision-making tasks.
- **`inhib.ipynb`**: Investigates the impact of inhibition on neural dynamics.
- **`plotting.py`**: Helper functions for visualizing results and simulations.
- **`recur_excit.ipynb`**: Studies the role of recurrent excitation in decision-making dynamics.
- **`role_inhib.ipynb`**: Focuses on the role of inhibition in decision-making processes.
- **`sim_dyn.py`**: Core simulation script for dynamic systems modeling.
- **`utils.py`**: Utility functions for preprocessing, computation, and support tasks.

### 2. `neuralAC__adaptive_learning/`
This folder contains code for implementing and analyzing a Neural Actor-Critic (NeuralAC) framework for adaptive learning.

#### Files:
- **`baselines.ipynb`**: Compares the performance of over wE.
- **`environment.py`**: Defines the environment setup for reinforcement learning tasks.
- **`experiment.py`**: Contains code for running experiments and simulations.
- **`neural_actor_critic.py`**: Implements the Neural Actor-Critic algorithm.
- **`plotting.py`**: Visualization scripts specific to adaptive learning experiments.
- **`role_inhib_strong_recur.ipynb`**: Examines the role of inhibition in recurrent networks with high precision.
- **`role_inhib_weak_recur.ipynb`**: Examines the role of inhibition in recurrent networks with low precision.
- **`utils.py`**: Common utility functions used across experiments.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Recommended packages:
    - `numpy`
    - `matplotlib`
    - `scipy`
    - `jupyter`
    - `pandas`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd EL_dynamics
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code
1. Navigate to the desired folder:
   ```bash
   cd mean_field__decision_making
   # or
   cd neuralAC__adaptive_learning
   ```
2. Open the notebooks in Jupyter:
   ```bash
   jupyter notebook <notebook_name>.ipynb
   ```

## Key Features
- Simulations of mean-field neural dynamics for decision-making.
- Analysis of recurrent excitation and inhibition mechanisms.
- Neural Actor-Critic implementation for adaptive learning.
- Visualization tools for understanding results.


