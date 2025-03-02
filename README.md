# CNN with Reinforcement Learning

This project implements a system that uses Reinforcement Learning (RL) to optimize CNN hyperparameters dynamically during training. The RL agent learns when and how to intervene in the CNN training process to improve performance.

## Overview

The system uses Proximal Policy Optimization (PPO) from Stable Baselines3 to make intelligent decisions about hyperparameter adjustments. The agent optimizes parameters like learning rate, dropout rate, weight decay, and optimizer selection based on observed training metrics.

## Features

- **Dynamic Hyperparameter Optimization**: Uses RL to adjust hyperparameters during CNN training
- **Adaptive Intervention**: Learns when to intervene in training and when to let the default process continue
- **Performance-Based Training Schedule**: Adjusts its own training frequency based on performance
- **Reward Normalization**: Uses normalized rewards for stable training
- **Model Persistence**: Saves the best performing models and can resume training
- **Customizable Thresholds**: Dynamically adjusts intervention thresholds based on performance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CNN-with-RL.git
cd CNN-with-RL

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Required Dependencies

- Python 3.7+
- PyTorch
- Stable Baselines3
- NumPy
- Matplotlib (for visualization)

## Project Structure

```
CNN-with-RL/
├── src/
│   ├── models/
│   │   ├── rl_agent.py           # RL-based hyperparameter optimizer
│   │   ├── cnn_model.py          # CNN implementation
│   │   └── environment.py        # RL environment for hyperparameter space
│   ├── utils/
│   │   ├── data_loader.py        # Data preprocessing utilities
│   │   └── visualization.py      # Training visualization tools
│   └── train.py                  # Main training script
├── notebooks/                    # Experimental Jupyter notebooks
├── logs/                         # Training logs
│   └── tensorboard/              # Tensorboard logs
├── models/                       # Saved models
│   └── rl_brains/                # Saved RL agent models
├── tests/                        # Unit tests
├── data/                         # Dataset storage
├── configs/                      # Configuration files
└── README.md                     # This file
```

## Usage

```python
from src.models.rl_agent import HyperParameterOptimizer
from your_environment import YourEnvironment

# Create RL environment
env = YourEnvironment()

# Configuration for the RL agent
config = {
    'learning_rate': 3e-4,
    'n_steps': 1024,
    'batch_size': 64,
    'brain_save_dir': 'models/rl_brains'
}

# Initialize optimizer
optimizer = HyperParameterOptimizer(env, config)

# Training loop
for epoch in range(num_epochs):
    # Get current CNN state/metrics as observation
    observation = get_current_state()
    
    # Get hyperparameter suggestion from RL agent
    hyperparams = optimizer.optimize_hyperparameters(observation)
    
    if hyperparams:
        # Apply suggested hyperparameters to CNN
        apply_hyperparameters(cnn_model, hyperparams)
        intervened = True
    else:
        intervened = False
    
    # Train CNN for one epoch
    train_cnn_epoch()
    
    # Get new observation and reward
    new_observation = get_current_state()
    reward = calculate_performance_improvement()
    
    # Train RL agent based on outcome
    optimizer.learn_from_intervention(new_observation, reward, intervened)

# Save the trained RL agent
optimizer.save_brain()
```

## Customization

You can customize the behavior of the RL agent by modifying the configuration parameters:

```python
config = {
    'learning_rate': 3e-4,           # Base learning rate for RL agent
    'n_steps': 1024,                 # Steps per update
    'batch_size': 64,                # Minibatch size
    'gamma': 0.99,                   # Discount factor
    'train_frequency_min': 5,        # Minimum training frequency
    'train_frequency_max': 25,       # Maximum training frequency
    'training_timesteps_min': 5000,  # Minimum training timesteps
    'training_timesteps_max': 10000, # Maximum training timesteps
    'intervention_threshold': 0.4    # Initial threshold for intervention
}
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

