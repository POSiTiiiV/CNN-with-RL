import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import torch.nn as nn

class CNNHyperparamEnv(gym.Env):
    """
    Custom Gym environment for CNN hyperparameter optimization
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, cnn_trainer, config):
        super(CNNHyperparamEnv, self).__init__()
        self.cnn_trainer = cnn_trainer
        self.config = config
        
        # Define action space (hyperparameters to optimize)
        # For example: learning rate, dropout rate, filter sizes
        self.action_space = spaces.Dict({
            'learning_rate': spaces.Discrete(10),  # 10 discrete levels mapped to a range
            'dropout_rate': spaces.Discrete(10),   # 10 discrete levels mapped to a range
            'conv1_filters': spaces.Discrete(8),   # From 16 to 128 filters in steps
            'conv2_filters': spaces.Discrete(8),   # From 32 to 256 filters in steps
        })
        
        # Define observation space
        # This could include current model performance, parameter counts, etc.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        
        self.current_hyperparams = {}
        self.current_val_acc = 0
        self.best_val_acc = 0
        self.episode_step = 0
        self.max_steps = config.get('max_steps_per_episode', 5)
        
    def _get_observation(self):
        """Create observation based on current state"""
        # Features could include:
        # - Normalized validation accuracy
        # - Normalized training accuracy
        # - Normalized loss values
        # - Normalized hyperparameter values
        # - Training progress (epoch/max_epoch)
        
        observation = np.zeros(10, dtype=np.float32)
        
        # Fill in some example values
        if hasattr(self.cnn_trainer, 'history') and self.cnn_trainer.history:
            observation[0] = self.current_val_acc  # Normalized val accuracy
            observation[1] = self.cnn_trainer.history['train_acc'][-1] if self.cnn_trainer.history['train_acc'] else 0  # Normalized train accuracy
            observation[2] = min(1.0, self.cnn_trainer.history['val_loss'][-1]/10) if self.cnn_trainer.history['val_loss'] else 0.5  # Normalized val loss
        
        # Normalized hyperparameters
        if self.current_hyperparams:
            observation[3] = self.current_hyperparams.get('learning_rate', 0.001) / 0.01
            observation[4] = self.current_hyperparams.get('dropout_rate', 0.5)
            observation[5] = self.current_hyperparams.get('conv1_filters', 32) / 128
            observation[6] = self.current_hyperparams.get('conv2_filters', 64) / 256
        
        observation[7] = self.episode_step / self.max_steps  # Progress in episode
        observation[8] = self.best_val_acc  # Best accuracy so far
        observation[9] = self.current_val_acc / max(self.best_val_acc, 0.001)  # Relative improvement
        
        return observation
    
    def _action_to_hyperparams(self, action):
        """Convert discrete action to actual hyperparameter values"""
        # Map discrete action indices to continuous hyperparameter values
        learning_rates = np.logspace(-4, -2, 10)  # From 0.0001 to 0.01
        dropout_rates = np.linspace(0.1, 0.9, 10)  # From 0.1 to 0.9
        conv1_filters = [16, 24, 32, 48, 64, 80, 96, 128]
        conv2_filters = [32, 48, 64, 96, 128, 160, 192, 256]
        
        hyperparams = {
            'learning_rate': learning_rates[action['learning_rate']],
            'dropout_rate': dropout_rates[action['dropout_rate']],
            'conv1_filters': conv1_filters[action['conv1_filters']],
            'conv2_filters': conv2_filters[action['conv2_filters']],
        }
        
        return hyperparams
    
    def _apply_hyperparameters(self, hyperparams):
        """Apply hyperparameters to the CNN model and trainer"""
        # Update optimizer learning rate
        for param_group in self.cnn_trainer.optimizer.param_groups:
            param_group['lr'] = hyperparams['learning_rate']
        
        # Update model architecture as needed
        # This requires modifications to the CNN model structure
        self.cnn_trainer.model.update_hyperparams(hyperparams)
    
    def step(self, action):
        """Take a step by applying hyperparameters and evaluating the model"""
        self.episode_step += 1
        
        # Convert action to hyperparameters
        self.current_hyperparams = self._action_to_hyperparams(action)
        
        # Apply hyperparameters to the model
        self._apply_hyperparameters(self.current_hyperparams)
        
        # Train for a few epochs
        self.cnn_trainer.train()
        
        # Get validation accuracy
        _, self.current_val_acc = self.cnn_trainer.evaluate()
        
        # Calculate reward based on validation accuracy improvement
        if self.current_val_acc > self.best_val_acc:
            # Reward for improving best accuracy
            reward = 10 * (self.current_val_acc - self.best_val_acc)
            self.best_val_acc = self.current_val_acc
        else:
            # Small negative reward for not improving
            reward = -0.1
        
        # Check if episode should terminate
        done = self.episode_step >= self.max_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Information dictionary
        info = {
            'val_acc': self.current_val_acc,
            'hyperparams': self.current_hyperparams,
            'best_val_acc': self.best_val_acc
        }
        
        return observation, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment to start a new episode"""
        super().reset(seed=seed)
        
        # Reset episode counter
        self.episode_step = 0
        
        # Reset current hyperparameters to defaults
        self.current_hyperparams = {
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'conv1_filters': 32,
            'conv2_filters': 64
        }
        
        # Apply default hyperparameters
        self._apply_hyperparameters(self.current_hyperparams)
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def render(self, mode='human'):
        """Render the environment"""
        print(f"Step: {self.episode_step}")
        print(f"Hyperparameters: {self.current_hyperparams}")
        print(f"Current validation accuracy: {self.current_val_acc:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
