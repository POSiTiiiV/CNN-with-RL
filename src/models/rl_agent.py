import os
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EvaluationCallback(BaseCallback):
    """
    Callback for evaluating the agent's performance during training.
    """
    def __init__(self, eval_freq=1000, verbose=1):
        super(EvaluationCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf

    def _on_step(self):
        """
        Called at each step of training.
        """
        if self.n_calls % self.eval_freq == 0:
            logger.info(f"EvaluationCallback step {self.n_calls}, "
                        f"mean reward: {self.model.ep_info_buffer.mean():.2f}")
            
            # Save best model
            if self.model.ep_info_buffer.mean() > self.best_mean_reward:
                self.best_mean_reward = self.model.ep_info_buffer.mean()
                self.model.save("best_model")
                logger.info(f"Saved new best model with mean reward: {self.best_mean_reward:.2f}")
                
        return True

class TimestepLoggingCallback(BaseCallback):
    """
    Custom callback for logging the number of timesteps done/remaining.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(TimestepLoggingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Log the number of timesteps done/remaining
        timesteps_done = self.num_timesteps
        timesteps_remaining = self.total_timesteps - timesteps_done
        logger.info(f"Timesteps done: {timesteps_done}, Timesteps remaining: {timesteps_remaining}")
        return True

class HyperParameterOptimizer:
    """
    RL-based hyperparameter optimizer for CNN models.
    Uses Proximal Policy Optimization to learn optimal hyperparameter settings.
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Extract configuration
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.n_steps = config.get('n_steps', 1024)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.normalize_advantage = config.get('normalize_advantage', True)
        self.brain_save_dir = config.get('brain_save_dir', 'models/rl_brains')

        # Dynamic threshold for intervention 
        self.intervention_threshold = config.get('intervention_threshold', 0.4)
        self.recent_intervention_scores = deque(maxlen=20)
        
        # Performance tracking for dynamic training schedule
        self.performance_history = deque(maxlen=10)
        self.reward_history = deque(maxlen=100)  # For normalizing rewards
        self.train_decision_counter = 0
        self.best_reward = float('-inf')  # Track best reward for saving best model

        # Dynamic training parameters
        # Start with lower frequency but adjust based on performance
        self.train_frequency_min = config.get('train_frequency_min', 5) 
        self.train_frequency_max = config.get('train_frequency_max', 25)
        self.train_frequency = self.train_frequency_min
        
        # Increased training timesteps as suggested (5000-10000)
        self.training_timesteps_min = config.get('training_timesteps_min', 5000)  # Minimum timesteps
        self.training_timesteps_max = config.get('training_timesteps_max', 10000) # Maximum timesteps
        self.training_timesteps = self.training_timesteps_min
        self.training_step = 0

        # Ensure brain save directory exists
        os.makedirs(self.brain_save_dir, exist_ok=True)

        # Initialize PPO agent with learning rate scheduler
        logger.info(f"Initializing PPO agent with learning rate scheduler")

        # IMPROVEMENT 2: Use a learning rate scheduler
        def lr_schedule(progress_remaining):
            """Linear learning rate decay"""
            return self.learning_rate * progress_remaining  # Decays from initial_lr to 0
        
        self.agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,  # Use scheduler instead of constant
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            normalize_advantage=self.normalize_advantage,
            ent_coef=config.get('ent_coef', 0.01),
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
            device="cpu"
        )

        self.eval_callback = EvaluationCallback(eval_freq=1000)
        logger.info("HyperParameterOptimizer initialized with improved parameters")

    def _update_dynamic_parameters(self, reward):
        """
        Update dynamic parameters based on recent performance.
        
        Args:
            reward (float): Most recent reward received
        """
        # Store performance data
        self.performance_history.append(reward)
        
        # Only adjust parameters when we have enough history
        if len(self.performance_history) >= 5:
            recent_rewards = list(self.performance_history)
            
            # Check if rewards are improving or declining
            is_improving = recent_rewards[-1] > np.mean(recent_rewards[:-1])
            
            # Adjust training frequency - train more often when performance is worse
            if is_improving:
                # If improving, we can train less frequently (save compute)
                self.train_frequency = min(self.train_frequency_max, 
                                          self.train_frequency + 2)
            else:
                # If not improving, train more frequently
                self.train_frequency = max(self.train_frequency_min, 
                                          self.train_frequency - 3)
            
            # Adjust training timesteps - train more steps when performance is worse
            if is_improving:
                # If improving, we can train fewer timesteps
                self.training_timesteps = max(self.training_timesteps_min,
                                            self.training_timesteps - 500)
            else:
                # If not improving, train with more timesteps
                self.training_timesteps = min(self.training_timesteps_max, 
                                            self.training_timesteps + 1000)
            
            # Adjust intervention threshold based on recent intervention decisions
            if self.recent_intervention_scores:
                # Calculate variance in decisions
                decision_variance = np.var(list(self.recent_intervention_scores))
                
                # If decisions are very consistent, adjust threshold to promote some exploration
                if decision_variance < 0.05:  # Low variance means consistent decisions
                    # Slightly lower threshold to encourage occasional interventions
                    self.intervention_threshold = max(0.2, self.intervention_threshold - 0.05)
                elif decision_variance > 0.2:  # High variance means too much flip-flopping
                    # Slightly raise threshold for more stability
                    self.intervention_threshold = min(0.6, self.intervention_threshold + 0.03)
                    
            logger.debug(f"Dynamic parameters updated - Train frequency: {self.train_frequency}, " +
                        f"Timesteps: {self.training_timesteps}, " +
                        f"Threshold: {self.intervention_threshold:.2f}")

    def optimize_hyperparameters(self, observation):
        """
        Decide whether to intervene based on the RL agent's policy.
        """
        self.last_observation = observation.copy()

        try:
            # Ensure observation matches expected shape
            if observation.shape != self.env.observation_space.shape:
                observation = np.resize(observation, self.env.observation_space.shape)

            # Get action from PPO
            action, _states = self.agent.predict(observation.reshape(1, -1))

            self.last_action = action[0].copy()
            intervention_score = action[0][0] if isinstance(action[0], np.ndarray) else action[0]
            
            # Track past intervention scores
            self.recent_intervention_scores.append(intervention_score)

            # Use the dynamic threshold for intervention decision
            logger.debug(f"Intervention score: {intervention_score:.4f} (threshold: {self.intervention_threshold:.4f})")

            if intervention_score < self.intervention_threshold:
                logger.info("RL agent decided not to intervene")
                return None

            # IMPROVEMENT 1: Fix potential KeyError in action_to_hp_dict
            try:
                hyperparams = self.env.action_to_hp_dict(action[0])
                # Validate hyperparams to ensure all required keys exist
                required_keys = ['learning_rate', 'dropout_rate', 'weight_decay', 'optimizer_type']
                if not all(key in hyperparams for key in required_keys):
                    logger.warning(f"Missing required keys in hyperparams: {hyperparams}")
                    # Provide defaults for any missing keys
                    defaults = {
                        'learning_rate': 0.001,
                        'dropout_rate': 0.5,
                        'weight_decay': 1e-4,
                        'optimizer_type': 'adam'
                    }
                    for key in required_keys:
                        if key not in hyperparams:
                            hyperparams[key] = defaults[key]
            except Exception as e:
                logger.error(f"Error converting action to hyperparams: {e}")
                return None  # Return None instead of crashing
                
            logger.info(f"Intervening with hyperparameters: {hyperparams}")
            return hyperparams

        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            return None

    def learn_from_intervention(self, new_observation, reward, intervened=True):
        """
        Train the RL agent based on the outcome of the last intervention.
        """
        if self.last_observation is None or self.last_action is None:
            logger.warning("Cannot train RL agent: no previous action stored")
            return False

        try:
            # Reward shaping
            if not intervened:
                reward = 0.1  # Base reward for stability
                if len(self.recent_intervention_scores) > 1:
                    recent_change = self.recent_intervention_scores[-1] - self.recent_intervention_scores[-2]
                    if recent_change < 0:
                        reward += 0.2  # Extra reward for avoiding bad interventions
            else:
                reward -= 0.05 * len(self.recent_intervention_scores)  # Penalty for frequent interventions
            
            # IMPROVEMENT 4: Normalize rewards
            self.reward_history.append(reward)
            if len(self.reward_history) > 1:
                mean_reward = np.mean(self.reward_history)
                std_reward = np.std(self.reward_history) + 1e-5  # Avoid division by zero
                normalized_reward = (reward - mean_reward) / std_reward
                
                # Cap normalized reward to avoid extreme values
                normalized_reward = max(min(normalized_reward, 5.0), -5.0)
                logger.info(f"Original reward: {reward:.4f}, Normalized: {normalized_reward:.4f}")
                reward = normalized_reward

            # IMPROVEMENT 3: Save best model when performance improves
            if reward > self.best_reward:
                self.best_reward = reward
                best_model_path = os.path.join(self.brain_save_dir, "best_rl_brain.zip")
                self.save_brain(best_model_path)
                logger.info(f"New best model saved with reward: {reward:.4f}")
            
            # Update dynamic parameters based on performance
            self._update_dynamic_parameters(reward)

            # Increment training step and decision counter
            self.training_step += 1
            self.train_decision_counter += 1

            # Update environment state if needed
            if hasattr(self.env, 'update_state'):
                self.env.update_state(new_observation)

            # Use dynamic training schedule based on performance
            if self.train_decision_counter >= self.train_frequency:
                self.train_decision_counter = 0  # Reset counter
                
                logger.info(f"Training PPO agent at step {self.training_step} for {self.training_timesteps} timesteps")
                
                # Create and use the custom callback for logging
                timestep_logging_callback = TimestepLoggingCallback(total_timesteps=self.training_timesteps)
                
                # Train the agent with updated parameters
                self.agent.learn(
                    total_timesteps=self.training_timesteps, 
                    reset_num_timesteps=False,
                    callback=timestep_logging_callback
                )
                
                # Log progress after training
                progress = min(1.0, self.training_step / 1000)
                current_lr = self.learning_rate * (1 - progress)
                logger.info(f"Completed training. Current LR: {current_lr:.6f}, Progress: {progress:.2f}")

            return True

        except Exception as e:
            logger.error(f"Error during RL agent training: {str(e)}")
            return False

    def save_brain(self, filepath=None):
        """
        Save the RL agent.
        """
        try:
            if filepath is None:
                import time
                filepath = os.path.join(self.brain_save_dir, f"rl_brain_{int(time.time())}.zip")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.agent.save(filepath)
            
            # Save additional metadata
            metadata = {
                'best_reward': self.best_reward,
                'training_step': self.training_step,
                'intervention_threshold': self.intervention_threshold
            }
            
            # Save metadata in a JSON file alongside the model
            metadata_path = filepath.replace('.zip', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved RL agent to {filepath} with metadata")
            return filepath
        except Exception as e:
            logger.error(f"Error saving RL agent: {str(e)}")
            return None

    def load_brain(self, filepath):
        """
        Load the RL agent with metadata if available.
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Brain file not found: {filepath}")
                return False
                
            self.agent = PPO.load(filepath, env=self.env, device="cpu")
            
            # Try to load metadata if available
            metadata_path = filepath.replace('.zip', '_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Restore state from metadata
                self.best_reward = metadata.get('best_reward', self.best_reward)
                self.training_step = metadata.get('training_step', self.training_step)
                self.intervention_threshold = metadata.get('intervention_threshold', self.intervention_threshold)
                
                logger.info(f"Restored agent state from metadata: best_reward={self.best_reward:.4f}, "
                           f"training_step={self.training_step}")
            
            logger.info(f"Loaded RL agent from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL agent: {str(e)}")
            return False
