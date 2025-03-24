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
- **Unified Training Workflow**: Streamlined architecture that uses consistent training methods for both standard training and RL intervention

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
- Gymnasium 
- Matplotlib (for visualization)
- Weights & Biases (optional, for experiment tracking)

## Project Structure

```
CNN-with-RL/
├── src/
│   ├── models/
│   │   ├── cnn.py               # CNN model implementation
│   │   └── rl_agent.py          # RL-based hyperparameter optimizer
│   ├── trainers/
│   │   ├── cnn_trainer.py       # CNN training logic
│   │   └── trainer.py           # Main training coordinator
│   ├── envs/
│   │   └── hpo_env.py           # RL environment for hyperparameter optimization
│   ├── data_loaders/
│   │   └── data_loader.py       # Data loading and preprocessing utilities
│   └── utils/
│       └── utils.py             # Helper functions for observations and metrics
├── configs/                     # Configuration files
│   └── default.yaml             # Default configuration
├── logs/                        # Training logs
├── models/                      # Saved models
│   └── rl_brains/               # Saved RL agent models
├── training_history/            # Training history JSON files
├── main.py                      # Main entry point
└── README.md                    # This file
```

## Training Workflow

The system implements a unified training workflow that combines CNN training with RL-based hyperparameter optimization:

1. **CNN Training Phase**: The CNN is trained using standard methods, while the RL agent observes the training process.

2. **Performance Monitoring**: During training, the system continuously monitors metrics like validation accuracy, loss trends, and training stability.

3. **Intervention Decisions**: When the RL agent detects stagnation or sub-optimal performance, it decides whether to intervene.

4. **Hyperparameter Adjustment**: If intervention is chosen, the RL agent selects new hyperparameters to improve performance.

5. **Feedback Loop**: The effects of interventions (or non-interventions) are tracked, and rewards are calculated based on subsequent performance changes.

6. **RL Training**: The RL agent learns from these rewards to improve its intervention policy over time.

## Usage

### Basic Usage

Run the main training script with default parameters:

```bash
python main.py
```

### Advanced Options

```bash
# Specify configuration file
python main.py --config configs/custom_config.yaml

# Set the data directory
python main.py --data-dir data/custom_dataset

# Override number of training epochs
python main.py --epochs 100

# Specify batch size
python main.py --batch-size 32

# Enable/disable Weights & Biases logging
python main.py --wandb        # Enable WandB
python main.py --no-wandb     # Disable WandB

# Resume training from checkpoint
python main.py --resume logs/checkpoints/checkpoint_epoch_50.pt

# Load pre-trained RL agent
python main.py --rl-brain models/rl_brains/trained_brain.zip

# Specify training device
python main.py --device cuda:0   # Use first GPU
python main.py --device cpu      # Use CPU
```

## System Components

### CNNTrainer

Responsible for training and evaluating the CNN model using standard deep learning techniques.

### HPOEnvironment

Provides a Gymnasium-compatible environment for the RL agent to interact with. It uses the underlying CNNTrainer to apply hyperparameter changes and get performance metrics.

### HyperParameterOptimizer

RL agent that learns when to intervene and what hyperparameter changes to make. Uses PPO to optimize its policy based on performance rewards.

### ModelTrainer

Coordinates the overall training process, managing the interaction between the CNN trainer and RL optimizer. It decides when to let the RL agent intervene in CNN training.

### Configuration

The system behavior can be customized through the configuration file:

```yaml
training:
  max_epochs: 100
  batch_size: 32
  early_stopping_patience: 15
  min_epochs_before_intervention: 10
  intervention_frequency: 5

rl:
  learning_rate: 3e-4
  n_steps: 1024
  batch_size: 64
  intervention_threshold: 0.6
  min_intervention_threshold: 0.3
  max_intervention_threshold: 0.8

env:
  max_steps_per_episode: 10
  epochs_per_step: 1
  reward_scaling: 10.0
  exploration_bonus: 0.5
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

