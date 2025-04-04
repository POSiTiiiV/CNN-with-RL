import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import copy
import os
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
from io import StringIO
from ..utils.utils import (
    create_observation, 
    enhance_observation_with_trends, 
    calculate_performance_trends, 
    get_hyperparams_hash
)

# Create a logger for rendering tables
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
        self.patience = self.config.get('patience', 6)  # Default to 10 steps
        self.intervention_frequency = config.get('intervention_frequency', 5)
        self.stagnation_epochs = 3
        self.stagnation_threshold = self.config.get('stagnation_threshold', 0.01)  # Default to 0.01
        
        # Save RL brain configuration
        self.brain_save_dir = config.get('brain_save_dir', 'models/rl_brains')
        self.brain_save_steps = config.get('brain_save_steps', 50)
        self.last_save_step = 0
        self.total_steps = 0
        self.best_episode_reward = -float('inf')
        os.makedirs(self.brain_save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.current_step = 0
        self.current_hyperparams = {}
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_train_acc = 0.0  # Add tracking for best training accuracy
        self.best_train_loss = float('inf')  # Add tracking for best training loss
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
        self.val_loss_history = deque(maxlen=self.stagnation_epochs)
        
        # Define the action space for hyperparameter tuning
        self.action_dims = [
            10,  # learning_rate: 10 options
            9,   # dropout_rate: 9 options (0.1 to 0.9)
            8,   # weight_decay: 8 options
            3,   # optimizer_type: 3 options
            3    # fc_config: 3 different configurations
        ]
        self.action_space = spaces.MultiDiscrete(self.action_dims)
        
        # Define the observation space
        self.observation_space = spaces.Dict({
            'metrics': spaces.Box(low=0.0, high=1.0, shape=(18,), dtype=np.float32),
            'hyperparams': spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        
        # Create mapping from action indices to hyperparameter values
        self._create_action_mappings()

        # Introduce a minimum training period before allowing hyperparameter updates
        self.min_training_steps = self.config.get('min_training_steps', 10)  # Default to 10 steps
        self.steps_since_last_update = 0  # Track steps since the last hyperparameter update
        
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
            [512, 256],       # Small  
            [1024, 512],      # Medium  
            [1024, 512, 256]  # Complex  
        ]

    def action_to_hp_dict(self, action):
        """Convert MultiDiscrete action to hyperparameter dictionary"""
        if isinstance(action, np.ndarray):
            action = action.tolist()
        return {
            'learning_rate': float(self.lr_values[action[0]]),
            'dropout_rate': float(self.dropout_values[action[1]]),
            'weight_decay': float(self.wd_values[action[2]]),
            'optimizer_type': self.optimizer_types[action[3]],
            'fc_layers': self.fc_configs[action[4]]
        }
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Create an observation from the current state.
        
        Returns:
            Dict[str, np.ndarray]: The current state observation
        """
        return create_observation({
            'history': self.history,
            'current_hyperparams': self.current_hyperparams,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'no_improvement_count': self.no_improvement_count,
            'patience': self.patience
        })

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment by applying new hyperparameters and training the CNN.
        
        Args:
            action: The action to take (hyperparameter changes)
            
        Returns:
            Tuple containing:
                observation (Dict[str, np.ndarray]): The new observation
                reward (float): The reward for the action
                terminated (bool): Whether the episode is done
                truncated (bool): Whether the episode is truncated
                info (dict): Additional information
        """
        self.current_step += 1
        self.total_steps += 1

        # Track loss history to check stagnation
        if self.history['val_loss']:
            self.val_loss_history.append(self.history['val_loss'][-1])
        
        # Check for stagnation
        stagnation_detected = (len(self.val_loss_history) == self.stagnation_epochs and
                              max(self.val_loss_history) - min(self.val_loss_history) < self.stagnation_threshold)
        
        if stagnation_detected:
            logger.info(f"Stagnation detected: loss diff {max(self.val_loss_history) - min(self.val_loss_history):.6f} < threshold {self.stagnation_threshold}")

        hyperparams_changed = False
        # Increment steps since last update
        self.steps_since_last_update += 1

        # Initialize is_new_config to False to ensure it is always defined
        is_new_config = False

        # Ensure RL agent explores new hyperparameters
        # Modify the logic to encourage exploration
        if stagnation_detected and self.steps_since_last_update >= self.min_training_steps:
            # Convert action to hyperparameters
            new_hyperparams = self.action_to_hp_dict(action)

            # Use centralized utility to check if hyperparams are new
            param_hash = get_hyperparams_hash(new_hyperparams)
            is_new_config = param_hash not in self.explored_configs

            # Check if they're actually different from current hyperparams
            if is_new_config:
                # Compare with current hyperparameters
                current_hash = get_hyperparams_hash(self.current_hyperparams) if self.current_hyperparams else None
                if current_hash != param_hash:
                    self.explored_configs.add(param_hash)
                    logger.info("Exploring new hyperparameters:")
                    for key, value in new_hyperparams.items():
                        logger.info(f"  - {key}: {value}")

                    self.current_hyperparams = new_hyperparams
                    self.cnn_trainer.model.update_hyperparams(self.current_hyperparams)
                    self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()
                    hyperparams_changed = True
                    self.steps_since_last_update = 0  # Reset the counter after an update
                else:
                    logger.debug("Generated hyperparameters are equivalent to current ones, not counted as new")

        # Remove functionality to skip training and validation
        # Always perform training and validation regardless of hyperparameter changes
        hyperparams_changed = True  # Force training and validation to always occur

        # Training and evaluation can be separate operations
        if hyperparams_changed:
            # Train for the specified number of epochs using the CNNTrainer
            train_metrics = None
            for _ in range(self.epochs_per_step):
                # Use CNNTrainer's train_epoch method
                train_metrics = self.cnn_trainer.train_epoch()

            # Evaluate using CNNTrainer's evaluate method
            val_metrics = self.cnn_trainer.evaluate()

            # Extract metrics
            train_loss = train_metrics['loss']
            train_acc = train_metrics['accuracy']
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']

            # Log metrics details
            logger.info("Metrics after training:")
            logger.info(f"  - Train Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
            logger.info(f"  - Val Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

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
            
        # Update best training metrics
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss

        # Check if episode is done
        done = (self.current_step >= self.max_steps) or (self.no_improvement_count >= self.patience)
        
        # Log episode end status
        if done:
            if self.current_step >= self.max_steps:
                logger.info(f"Episode ended: Max steps ({self.max_steps}) reached")
            else:
                logger.info(f"Episode ended: No improvement for {self.no_improvement_count} steps (patience: {self.patience})")
            
            # Calculate and log total episode reward
            episode_reward = sum(self.history['rewards'][-self.current_step:])
            logger.info(f"Episode total reward: {episode_reward:.4f}")
        
        # Get next observation
        observation = self._get_observation()

        # Add trend data to observation
        trend_data = calculate_performance_trends({
            'history': self.history,
            'metric_window_size': self.config.get('metric_window_size', 5),
            'improvement_threshold': self.config.get('improvement_threshold', 0.002),
            'loss_stagnation_threshold': self.config.get('loss_stagnation_threshold', 0.003)
        })
        observation = enhance_observation_with_trends(observation, trend_data)

        # Save brain periodically during training
        if self.total_steps - self.last_save_step >= self.brain_save_steps:
            self.save_agent_brain()
            self.last_save_step = self.total_steps
        
        # Check if this is the best episode reward so far and save special brain if it is
        episode_reward = sum(self.history['rewards'][-self.current_step:])
        if done and episode_reward > self.best_episode_reward:
            self.best_episode_reward = episode_reward
            self.save_agent_brain(best=True)

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
            'is_new_config': is_new_config,
            'stagnation_detected': stagnation_detected,
            'hyperparams_changed': hyperparams_changed,
            'total_steps': self.total_steps,
            'episode_reward': episode_reward
        }

        # Add symbols and new lines to clearly distinguish between steps and episodes
        logger.info("=====================")
        logger.info(f"Step {self.current_step}/{self.max_steps}")
        logger.info("=====================\n")

        # At the end of an episode, add a separator
        if done:
            logger.info("=====================")
            logger.info("End of Episode")
            logger.info("=====================\n")

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
        # Base reward is a combination of validation accuracy and loss
        reward = math.log(1 + val_acc) - 0.1 * val_loss  

        # Bonus for improvement in validation accuracy
        improvement = 0
        loss_improvement = 0
        if val_acc > self.best_val_acc:
            improvement = val_acc - self.best_val_acc
            reward += self.reward_scaling * improvement
            logger.info(f"Accuracy improvement: +{improvement:.4f}")

        # Bonus for improvement in validation loss
        if val_loss < self.best_val_loss:
            loss_improvement = self.best_val_loss - val_loss
            reward += self.reward_scaling * loss_improvement
            logger.info(f"Loss improvement: -{loss_improvement:.4f}")

        # Exploration bonus (less weight compared to improvements)
        if is_new_config:
            exploration_reward = 0.1 * self.exploration_bonus
            reward += exploration_reward
            logger.info(f"Exploration bonus: +{exploration_reward:.4f}")

        # Penalty for no improvement
        if self.no_improvement_count > 0:
            penalty = 0.1 * self.no_improvement_count
            reward -= penalty
            logger.warning(f"No improvement penalty: -{penalty:.4f}")

        # Prevent negative rewards
        return max(0, reward)

    def render(self):
        """
        Render the current state of the environment with Rich formatting.
        """
        if not self.render_mode:
            return

        logger.info(f"HPO Environment - Step {self.current_step}/{self.max_steps}")
        
        # Replace Rich Table with plain logging for hyperparameters
        print('\n')
        logger.info("Current Hyperparameters:")
        print('\n')
        for name, value in self.current_hyperparams.items():
            if isinstance(value, float):
                logger.info(f"{name}: {value:.6f}")
            else:
                logger.info(f"{name}: {value}")

        # Replace Rich Table for performance metrics with plain logging
        if self.history['val_acc']:
            print('\n')
            logger.info("Performance Metrics:")
            print('\n')
            logger.info(f"Val Accuracy: Current: {self.history['val_acc'][-1]:.4f}, Best: {self.best_val_acc:.4f}")
            logger.info(f"Val Loss: Current: {self.history['val_loss'][-1]:.4f}, Best: {self.best_val_loss:.4f}")
            if len(self.history['train_acc']) > 0:
                logger.info(f"Train Accuracy: Current: {self.history['train_acc'][-1]:.4f}, Best: {self.best_train_acc:.4f}")
                logger.info(f"Train Loss: Current: {self.history['train_loss'][-1]:.4f}, Best: {self.best_train_loss:.4f}")
        
        # Display reward
        if self.history['rewards']:
            reward = self.history['rewards'][-1]
            reward_color = "green" if reward > 0 else "red"
            logger.info(f"Reward: {reward:.4f}")
        
        # Show no improvement counter
        improvement_status = "Good" if self.no_improvement_count < self.patience // 2 else "Concerning" if self.no_improvement_count < self.patience else "Critical"
        logger.info(f"Steps without improvement: {self.no_improvement_count}/{self.patience} ({improvement_status})")

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Seed for random number generator
            options: Additional options
            
        Returns:
            Tuple containing:
                observation (Dict[str, np.ndarray]): The initial observation
                info (dict): Additional information
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Check if the previous run was bad (reached patience limit without improvement)
        bad_previous_run = self.no_improvement_count >= self.patience
        
        # Reset training state
        self.current_step = 0
        self.no_improvement_count = 0
        self.val_loss_history.clear()
        
        # Get initial hyperparameters - either use default or get from options
        if options and 'hyperparams' in options:
            self.current_hyperparams = options['hyperparams']
            logger.info(f"Reset environment with provided hyperparameters: {get_hyperparams_hash(self.current_hyperparams)}")
        elif not self.current_hyperparams or bad_previous_run:  # Reset if not set or previous run was bad
            # Use default hyperparameters
            self.current_hyperparams = {
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'weight_decay': 0.0001,
                'optimizer_type': 'adam',
                'fc_layers': [1024, 512]
            }
            if bad_previous_run:
                logger.info(f"Previous run reached patience limit. Resetting hyperparameters.")
            logger.info(f"Reset environment with default hyperparameters: {get_hyperparams_hash(self.current_hyperparams)}")
        else:
            logger.info(f"Keeping existing hyperparameters during reset: {get_hyperparams_hash(self.current_hyperparams)}")
        
        # Apply hyperparameters to the model
        self.cnn_trainer.model.update_hyperparams(self.current_hyperparams)
        self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()
        
        # Reset best metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_hyperparams = copy.deepcopy(self.current_hyperparams)
        
        # Reset history
        self.history = {
            'val_acc': [],
            'val_loss': [],
            'train_acc': [],
            'train_loss': [],
            'hyperparams': [copy.deepcopy(self.current_hyperparams)],
            'rewards': []
        }
        
        # Get initial metrics to include in observation
        try:
            val_metrics = self.cnn_trainer.evaluate()
            if val_metrics['loss'] is not None:
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['loss'])
                
                # Update best metrics if needed
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_val_loss = val_metrics['loss']
        except Exception as e:
            logger.error(f"Error getting initial metrics: {e}")
            # Use placeholder values if evaluation fails
            self.history['val_acc'].append(0.0)
            self.history['val_loss'].append(1.0)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Add trend data
        if len(self.history['val_acc']) >= 2:
            trend_data = calculate_performance_trends({
                'history': self.history,
                'metric_window_size': self.config.get('metric_window_size', 5),
                'improvement_threshold': self.config.get('improvement_threshold', 0.002),
                'loss_stagnation_threshold': self.config.get('loss_stagnation_threshold', 0.003)
            })
            observation = enhance_observation_with_trends(observation, trend_data)
            
        # Info dictionary
        info = {
            'initial_hyperparams': self.current_hyperparams,
            'initial_val_acc': self.history['val_acc'][0] if self.history['val_acc'] else None,
            'initial_val_loss': self.history['val_loss'][0] if self.history['val_loss'] else None
        }
        
        if self.render_mode == 'human':
            print('\n')
            logger.info("Environment reset")
            print('\n')
            self.render()
            
        return observation, info

    def save_agent_brain(self, best=False) -> str:
        """
        Save the RL agent brain.
        
        Args:
            best: Whether to save as a "best" brain
            
        Returns:
            str: Path to the saved brain file
        """
        if not hasattr(self, 'agent') or not hasattr(self.agent, 'save'):
            logger.warning("Warning: Environment has no agent attribute with save method")
            return None
            
        try:
            timestamp = int(time.time())
            
            # Different naming for best brains
            if best:
                filename = "hpo_env_best_brain.zip"
                logger.info(f"Saving best RL brain with episode reward: {self.best_episode_reward:.4f}")
            else:
                filename = f"hpo_env_brain_step_{self.total_steps}.zip"
                logger.info(f"Saving periodic RL brain at step {self.total_steps}")
            
            # Full path
            brain_path = os.path.join(self.brain_save_dir, filename)
            
            # Save the brain
            logger.info(f"Saving brain to {brain_path}...")
            self.agent.save(brain_path)
            
            # Save metadata
            metadata = {
                "total_steps": self.total_steps,
                "timestamp": timestamp,
                "best_val_acc": float(self.best_val_acc),
                "best_val_loss": float(self.best_val_loss),
                "current_hyperparams": self.current_hyperparams,
                "best_hyperparams": self.best_hyperparams,
                "episode_reward": float(self.history['rewards'][-1]) if self.history['rewards'] else 0.0,
                "no_improvement_count": self.no_improvement_count,
                "explored_configs_count": len(self.explored_configs)
            }
            
            # Add best episode reward if it's a best brain save
            if best:
                metadata["best_episode_reward"] = float(self.best_episode_reward)
                
            # Save metadata to JSON file
            metadata_path = brain_path.replace(".zip", "_metadata.json")
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Brain saved to {brain_path}")
            return brain_path
            
        except Exception as e:
            logger.exception("Error saving RL brain")
            logger.error(f"Error saving RL brain: {str(e)}")
            return None

    def set_agent(self, agent):
        """
        Set the RL agent for this environment.
        
        Args:
            agent: The RL agent to use
        """
        self.agent = agent
        logger.info("RL agent set in HPO environment")
