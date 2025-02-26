
# CNN with Reinforcement Learning for Hyperparameter Optimization

This project implements a novel approach to training Convolutional Neural Networks (CNNs) by using Reinforcement Learning (RL) to optimize hyperparameters dynamically during the training process. The system is specifically tailored for medical image classification, particularly fundus images for diagnosing various eye conditions.

## Project Overview

Traditional hyperparameter optimization methods often require multiple training runs or pre-training validation. This project takes a different approach by using an RL agent that observes the training process and makes real-time adjustments to hyperparameters like learning rate, momentum, and batch size. This creates a more efficient training pipeline that can adapt to the specific characteristics of the model and dataset.

### Key Features

- **Dynamic Hyperparameter Optimization**: Uses reinforcement learning to adjust hyperparameters during CNN training
- **Transfer Learning**: Leverages pre-trained CNN architectures (ResNet, EfficientNet, etc.) for medical image classification
- **Custom Dataset Support**: Includes a data pipeline for fundus image datasets with 8 different disease classifications
- **Comprehensive Logging**: Integration with Weights & Biases (wandb) for experiment tracking
- **Flexible Configuration**: YAML-based configuration for easy experiment customization

## Project Structure

