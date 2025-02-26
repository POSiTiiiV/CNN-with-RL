import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class HPOEnvironment(gym.Env):
    """
    Environment for hyperparameter optimization of CNN models using RL.
    
    This environment allows an RL agent to tune hyperparameters of a CNN model
    and get rewards based on the model's validation performance.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, cnn_trainer, config, render_mode=None):
        super(HPOEnvironment, self).__init__()
        
        self.cnn_trainer = cnn_trainer
        self.config = config
        self.render_mode = render_mode
        
        # Extract configuration
        self.max_steps = config.get('max_steps_per_episode', 10)
        self.epochs_per_step = config.get('epochs_per_step', 1)
        self.reward_scaling = config.get('reward_scaling', 10.0)
        self.exploration_bonus = config.get('exploration_bonus', 0.5)
        self.patience = config.get('patience', 3)
        
        # Initialize tracking variables
        self.current_step = 0
        self.current_hyperparams = {}
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_hyperparams = {}
        self.history = {
            'val_acc': [],
            'val_loss': [],
            'train_acc': [],
            'train_loss': [],
            'hyperparams': [],
            'rewards': []
        }
        self.no_improvement_count = 0
        self.explored_configs = set()
        
        # Define the action space for hyperparameter tuning
        # Use MultiDiscrete instead of Dict for compatibility with Stable Baselines3
        self.action_dims = [
            10,  # learning_rate: 10 options
            9,   # dropout_rate: 9 options (0.1 to 0.9)
            8,   # weight_decay: 8 options
            3,   # optimizer_type: 3 options
            6    # fc_config: 6 different configurations
        ]
        self.action_space = spaces.MultiDiscrete(self.action_dims)
        
        # Define the observation space
        # This includes normalized metrics and hyperparameter values
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(15,),  # 15 features to track state
            dtype=np.float32
        )
        
        # Create mapping from action indices to hyperparameter values
        self._create_action_mappings()
        
    def _create_action_mappings(self):
        """Create mappings between action indices and actual hyperparameter values"""
        # Learning rate mapping (logarithmic scale from 1e-5 to 1e-1)
        self.lr_values = np.logspace(-5, -1, self.action_dims[0])
        
        # Dropout rate mapping (linear scale from 0.1 to 0.9)
        self.dropout_values = np.linspace(0.1, 0.9, self.action_dims[1])
        
        # Weight decay mapping (logarithmic scale from 1e-6 to 1e-2)
        self.wd_values = np.logspace(-6, -2, self.action_dims[2])
        
        # Optimizer type mapping
        self.optimizer_types = ['adam', 'sgd', 'adamw']
        
        # FC layer configurations mapping
        self.fc_configs = [
            [512, 256],
            [1024, 512],
            [2048, 1024],
            [512, 256, 128],
            [1024, 512, 256],
            [2048, 1024, 512]
        ]
        
    def action_to_hp_dict(self, action):
        """Convert MultiDiscrete action to hyperparameter dictionary"""
        return {
            'learning_rate': float(self.lr_values[action[0]]),
            'dropout_rate': float(self.dropout_values[action[1]]),
            'weight_decay': float(self.wd_values[action[2]]),
            'optimizer_type': self.optimizer_types[action[3]],
            'fc_config': self.fc_configs[action[4]]
        }
        
    def _get_observation(self) -> np.ndarray:
        """
        Create an observation from the current state.
        
        Returns:
            np.ndarray: The current state observation
        """
        observation = np.zeros(15, dtype=np.float32)
        
        # Current performance metrics
        if self.history['val_acc']:
            observation[0] = self.history['val_acc'][-1]  # Current val accuracy
            observation[1] = min(1.0, max(0.0, 1.0 - self.history['val_loss'][-1] / 10))  # Normalized val loss
            observation[2] = self.history['train_acc'][-1] if self.history['train_acc'] else 0.0
            observation[3] = min(1.0, max(0.0, 1.0 - self.history['train_loss'][-1] / 10))
            
        # Best performance so far
        observation[4] = self.best_val_acc
        observation[5] = min(1.0, max(0.0, 1.0 - self.best_val_loss / 10))
        
        # Current hyperparameters (normalized)
        if self.current_hyperparams:
            # Learning rate (log scale normalization)
            lr = self.current_hyperparams.get('learning_rate', 0.001)
            observation[6] = (np.log10(lr) + 5) / 3  # Log scale from 1e-5 to 1e-2
            
            # Weight decay (log scale normalization)
            wd = self.current_hyperparams.get('weight_decay', 1e-4)
            observation[7] = (np.log10(wd) + 6) / 4  # Log scale from 1e-6 to 1e-2
            
            # Dropout rate (linear scale)
            observation[8] = self.current_hyperparams.get('dropout_rate', 0.5)
            
            # Optimizer type (one-hot like encoding)
            opt_type = self.current_hyperparams.get('optimizer_type', 'adam')
            if opt_type == 'adam':
                observation[9] = 0.33
            elif opt_type == 'sgd':
                observation[10] = 0.33
            else:  # adamw
                observation[11] = 0.33
        
        # Episode progress
        observation[12] = self.current_step / self.max_steps
        
        # No improvement counter (normalized)
        observation[13] = self.no_improvement_count / self.patience
        
        # Relative improvement from last step
        if len(self.history['val_acc']) > 1:
            last_acc = self.history['val_acc'][-2]
            current_acc = self.history['val_acc'][-1]
            rel_improvement = (current_acc - last_acc) / max(0.01, last_acc)
            observation[14] = min(1.0, max(0.0, rel_improvement + 0.5))  # Scale to [0,1]
            
        return observation
        
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step by applying hyperparameters and evaluating the model.
        
        Args:
            action: Dictionary of actions from the RL agent
            
        Returns:
            Tuple containing:
                observation (np.ndarray): Current state
                reward (float): Reward for the action
                terminated (bool): Whether the episode is done
                truncated (bool): Whether the episode was truncated
                info (dict): Additional information
        """
        self.current_step += 1
        
        # Convert MultiDiscrete action to hyperparameter dictionary
        self.current_hyperparams = self.action_to_hp_dict(action)
        
        # Create a config hash to track exploration
        param_hash = self._get_param_hash(self.current_hyperparams)
        is_new_config = param_hash not in self.explored_configs
        self.explored_configs.add(param_hash)
        
        # Apply hyperparameters to the model
        self.cnn_trainer.model.update_hyperparams(self.current_hyperparams)
        
        # Update optimizer with new hyperparameters
        self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()
        
        # Train for defined number of epochs
        for _ in range(self.epochs_per_step):
            train_loss, train_acc = self.cnn_trainer.train_epoch()
        
        # Evaluate on validation set
        val_loss, val_acc = self.cnn_trainer.evaluate()
        
        # Update history
        self.history['val_acc'].append(val_acc)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['train_loss'].append(train_loss)
        self.history['hyperparams'].append(copy.deepcopy(self.current_hyperparams))
        
        # Calculate reward
        reward = self._calculate_reward(val_acc, val_loss, is_new_config)
        self.history['rewards'].append(reward)
        
        # Check for improvement
        improved = False
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_hyperparams = copy.deepcopy(self.current_hyperparams)
            self.no_improvement_count = 0
            improved = True
        else:
            self.no_improvement_count += 1
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps) or (self.no_improvement_count >= self.patience)
        
        # Get next observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'hyperparams': self.current_hyperparams,
            'best_hyperparams': self.best_hyperparams,
            'improved': improved,
            'steps_without_improvement': self.no_improvement_count,
            'is_new_config': is_new_config
        }
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
            
        return observation, reward, done, False, info
        
    def _calculate_reward(self, val_acc: float, val_loss: float, is_new_config: bool) -> float:
        """
        Calculate reward based on validation metrics.
        
        Args:
            val_acc: Validation accuracy
            val_loss: Validation loss
            is_new_config: Whether this is a previously unexplored configuration
            
        Returns:
            float: The reward value
        """
        # Base reward is proportional to validation accuracy
        reward = val_acc
        
        # Add bonus for improvement over best score
        if val_acc > self.best_val_acc:
            improvement = val_acc - self.best_val_acc
            reward += self.reward_scaling * improvement
        
        # Add bonus for exploring new configurations
        if is_new_config:
            reward += self.exploration_bonus
            
        # Add penalty for no improvement
        if self.no_improvement_count > 0:
            reward -= 0.1 * self.no_improvement_count
            
        return reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            Tuple containing:
                observation (np.ndarray): Initial observation
                info (dict): Additional information
        """
        super().reset(seed=seed)
        
        # Reset counters and trackers
        self.current_step = 0
        self.no_improvement_count = 0
        
        # Reset to default hyperparameters
        self.current_hyperparams = {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'dropout_rate': 0.5,
            'fc_layers': [512, 256],
            'optimizer_type': 'adam'
        }
        
        # Apply default hyperparameters
        self.cnn_trainer.model.update_hyperparams(self.current_hyperparams)
        self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()
        
        # Reset history
        if options and options.get('keep_history', False):
            # Just append a marker to separate episodes
            for key in self.history.keys():
                if key != 'hyperparams':
                    self.history[key].append(None)
        else:
            # Clear history completely
            self.history = {
                'val_acc': [],
                'val_loss': [],
                'train_acc': [],
                'train_loss': [],
                'hyperparams': [],
                'rewards': []
            }
            
            # Reset best metrics if not keeping history
            if not (options and options.get('keep_best', False)):
                self.best_val_acc = 0.0
                self.best_val_loss = float('inf')
                self.best_hyperparams = {}
        
        # Initial evaluation to get baseline metrics
        val_loss, val_acc = self.cnn_trainer.evaluate()
        train_loss = val_loss  # Placeholder until we train
        train_acc = val_acc    # Placeholder until we train
        
        # Update history with initial values
        self.history['val_acc'].append(val_acc)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['train_loss'].append(train_loss)
        self.history['hyperparams'].append(copy.deepcopy(self.current_hyperparams))
        self.history['rewards'].append(0.0)  # No reward for initial state
        
        # Update best metrics if this is truly the first evaluation
        if len(self.history['val_acc']) == 1 or options and not options.get('keep_best', False):
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_hyperparams = copy.deepcopy(self.current_hyperparams)
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'reset_type': 'full' if not (options and options.get('keep_history', False)) else 'episode'
        }
        
        return observation, info
        
    def _get_param_hash(self, hyperparams: Dict[str, Any]) -> str:
        """
        Create a hash string for a hyperparameter configuration to track explored configs.
        
        Args:
            hyperparams: Hyperparameter dictionary
            
        Returns:
            str: Hash string representing the configuration
        """
        # Round numerical values to avoid minor differences
        lr = round(hyperparams.get('learning_rate', 0), 6)
        wd = round(hyperparams.get('weight_decay', 0), 8)
        dr = round(hyperparams.get('dropout_rate', 0), 2)
        fc = '-'.join([str(x) for x in hyperparams.get('fc_layers', [])])
        opt = hyperparams.get('optimizer_type', '')
        
        return f"lr{lr}_wd{wd}_dr{dr}_fc{fc}_opt{opt}"
        
    def render(self):
        """
        Render the current state of the environment.
        """
        if not self.render_mode:
            return
            
        print("\n" + "="*50)
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Current hyperparameters:")
        for name, value in self.current_hyperparams.items():
            print(f"  {name}: {value}")
            
        if self.history['val_acc']:
            print(f"\nPerformance:")
            print(f"  Val Acc: {self.history['val_acc'][-1]:.4f}")
            print(f"  Val Loss: {self.history['val_loss'][-1]:.4f}")
            if len(self.history['train_acc']) > 0:
                print(f"  Train Acc: {self.history['train_acc'][-1]:.4f}")
                print(f"  Train Loss: {self.history['train_loss'][-1]:.4f}")
            
        print(f"\nBest so far:")
        print(f"  Val Acc: {self.best_val_acc:.4f}")
        print(f"  Val Loss: {self.best_val_loss:.4f}")
        
        if self.history['rewards']:
            print(f"\nReward: {self.history['rewards'][-1]:.4f}")
            
        print(f"Steps without improvement: {self.no_improvement_count}")
        print("="*50 + "\n")
