import os
import numpy as np
import logging
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from typing import Dict, Any, Optional, List

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

class LRScheduler:
    """Simple learning rate scheduler for PPO agent"""
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr

    def __call__(self, progress_remaining):
        """Linear learning rate decay"""
        return self.initial_lr * progress_remaining

class HyperParameterOptimizer:
    """
    RL-based hyperparameter optimizer for CNN models.
    Uses Proximal Policy Optimization to learn optimal hyperparameter settings.
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Extract core configuration parameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.n_steps = config.get('n_steps', 1024)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.gamma = config.get('gamma', 0.99)
        self.brain_save_dir = config.get('brain_save_dir', 'models/rl_brains')

        # Dynamic threshold for intervention 
        self.intervention_threshold = config.get('intervention_threshold', 0.6)
        self.recent_intervention_scores = deque(maxlen=20)
        
        # For tracking episodes when training the agent
        self.collected_episodes = []
        self.max_episodes_memory = config.get('max_episodes_memory', 20)
        self.min_episodes_for_training = config.get('min_episodes_for_training', 3)
        
        # Episodes persistence
        self.episodes_save_path = config.get('episodes_save_path', os.path.join('logs', 'rl_episodes.json'))
        self.auto_save_episodes = config.get('auto_save_episodes', True)
        self.auto_load_episodes = config.get('auto_load_episodes', True)
        
        # Intervention threshold boundaries
        self.min_intervention_threshold = config.get('min_intervention_threshold', 0.3)
        self.max_intervention_threshold = config.get('max_intervention_threshold', 0.8)
        
        # Training parameters
        self.training_timesteps_min = config.get('training_timesteps_min', 500)
        self.training_timesteps_max = config.get('training_timesteps_max', 10000)
        self.training_timesteps = self.training_timesteps_min
        self.training_step = 0
        
        # Ensure brain save directory exists
        os.makedirs(self.brain_save_dir, exist_ok=True)

        # Initialize PPO agent (always on CPU for stability regardless of GPU availability)
        logger.info(f"Initializing PPO agent on CPU")
        self.lr_scheduler = LRScheduler(self.learning_rate)
        self.agent = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=self.lr_scheduler,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            normalize_advantage=config.get('normalize_advantage', True),
            ent_coef=config.get('ent_coef', 0.01),
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
            device="cpu"  # Always use CPU for RL agent
        )

        # Store most recent action and state for learning
        self.last_state = None
        self.last_action = None

        # Load previously collected episodes if available and configured
        if self.auto_load_episodes:
            self._load_episodes()

        logger.info("HyperParameterOptimizer initialized on CPU")

    def optimize_hyperparameters(self, observation: Dict[str, np.ndarray], current_hyperparams: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Decide whether to intervene based on current model performance.
        If intervention score exceeds threshold, return new hyperparameters.
        
        Args:
            observation: Current state observation from environment
            current_hyperparams: Currently used hyperparameters (to avoid redundant changes)
            
        Returns:
            Optional[Dict[str, Any]]: New hyperparameters if intervention is needed, None otherwise
        """
        try:
            # Check for NaN values in observation and replace them
            for key, value in observation.items():
                if isinstance(value, np.ndarray) and np.isnan(value).any():
                    logger.warning(f"Found NaN values in observation key '{key}', replacing with zeros")
                    observation[key] = np.nan_to_num(value, nan=0.0)

            # Get action from PPO agent (intervention decision + hyperparameter changes)
            try:
                action, _states = self.agent.predict(observation)
            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    logger.warning(f"Numerical instability in prediction: {e}. Returning no intervention.")
                    return None
                raise e
            
            # Check for NaN values in the action
            if isinstance(action, np.ndarray) and np.isnan(action).any():
                logger.warning("Action contains NaN values, returning no intervention")
                return None
                
            # Extract intervention score (first action component)
            intervention_score = action[0][0] if isinstance(action[0], np.ndarray) else action[0]
            
            # Additional check for NaN in intervention score
            if np.isnan(intervention_score):
                logger.warning("Intervention score is NaN, returning no intervention")
                return None
                
            # Track recent scores
            self.recent_intervention_scores.append(intervention_score)
            
            logger.debug(f"Intervention score: {intervention_score:.4f} (threshold: {self.intervention_threshold:.4f})")
            
            # If score below threshold, don't intervene
            if intervention_score < self.intervention_threshold:
                logger.info(f"Intervention score {intervention_score:.4f} below threshold {self.intervention_threshold:.4f}, no intervention")
                return None 

            # Convert action to hyperparameters
            try:
                if isinstance(action[0], np.ndarray):
                    action = action[0].tolist()
                
                # Check for NaN values in action before conversion
                if any(np.isnan(x) if isinstance(x, (float, np.float32, np.float64)) else False for x in action):
                    logger.warning("NaN values in action, returning no intervention")
                    return None
                
                hyperparams = self.env.action_to_hp_dict(action)
                
                # Verify hyperparams don't contain NaN values
                for k, v in hyperparams.items():
                    if isinstance(v, (float, np.float32, np.float64)) and np.isnan(v):
                        logger.warning(f"NaN value in hyperparameter {k}, returning no intervention")
                        return None
            except Exception as e:
                logger.error(f"Error converting action to hyperparams: {e}")
                return None

            # Check if the new hyperparameters are sufficiently different from current ones
            if current_hyperparams is not None:
                has_significant_change = False
                for key, new_value in hyperparams.items():
                    if key in current_hyperparams:
                        current_value = current_hyperparams[key]
                        
                        # For numeric values, check percentage difference
                        if isinstance(new_value, (int, float)) and isinstance(current_value, (int, float)):
                            if current_value != 0:
                                # If change is less than 5%, consider it not significant
                                pct_diff = abs(new_value - current_value) / abs(current_value)
                                if pct_diff > 0.05:  # 5% threshold
                                    has_significant_change = True
                                    break
                            elif new_value != 0:  # Current is 0 but new is not
                                has_significant_change = True
                                break
                        # For non-numeric values, direct comparison
                        elif new_value != current_value:
                            has_significant_change = True
                            break
                    else:
                        # New parameter that wasn't in current set
                        has_significant_change = True
                        break
                        
                if not has_significant_change:
                    logger.info("Proposed hyperparameters too similar to current ones, skipping intervention")
                    return None
                    
                logger.info("Significant hyperparameter changes detected, proceeding with intervention")
    
            return hyperparams

        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            return None

    def learn_from_episode(self, episode_experiences: List[Dict]) -> bool:
        """
        Train the agent based on collected episode experiences in SARST format.
        
        Args:
            episode_experiences: List of experiences, each containing state, action, reward, next_state, done
            
        Returns:
            bool: Whether training was performed
        """
        try:
            if not episode_experiences:
                logger.warning("Empty episode experiences, skipping collection")
                return False
                
            # Store this episode's experiences
            self.collected_episodes.append(episode_experiences)
            logger.info(f"Collected episode with {len(episode_experiences)} experiences. " + 
                       f"Total episodes: {len(self.collected_episodes)}/{self.min_episodes_for_training}")
            
            # Keep memory limited
            if len(self.collected_episodes) > self.max_episodes_memory:
                self.collected_episodes = self.collected_episodes[-self.max_episodes_memory:]
            
            # Save episodes immediately after collection if configured
            if self.auto_save_episodes:
                self._save_episodes()
            
            # Only train if we have enough episodes
            if len(self.collected_episodes) < self.min_episodes_for_training:
                return False
                
            # When we have enough episodes, trigger the actual training
            return self._train_agent_with_collected_episodes()

        except Exception as e:
            logger.error(f"Error during RL agent episode collection: {str(e)}")
            return False
            
    def _train_agent_with_collected_episodes(self) -> bool:
        """
        Train the agent using all collected episodes with SARST format.
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        if not self.collected_episodes:
            logger.warning("No episodes to train on")
            return False
            
        # Extract all experiences from collected episodes
        all_rewards = []
        experience_count = 0
        
        # Count experiences and collect rewards for statistics
        for ep in self.collected_episodes:
            experience_count += len(ep)
            # Extract rewards from each experience tuple
            ep_rewards = [exp['reward'] for exp in ep if 'reward' in exp]
            all_rewards.extend(ep_rewards)
                
        if not all_rewards:
            logger.warning("No rewards found in collected episodes")
            return False
            
        # Log training statistics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        logger.info(f"Training PPO agent with {experience_count} experiences from {len(self.collected_episodes)} episodes")
        logger.info(f"Reward statistics: mean={mean_reward:.4f}, std={std_reward:.4f}, min={min(all_rewards):.4f}, max={max(all_rewards):.4f}")
        
        # Create a progress bar for training
        total_steps = self.training_timesteps
        progress_interval = max(1, total_steps // 20)  # Report progress ~20 times
        logger.info(f"Starting training for {total_steps} timesteps...")
        
        # Custom callback to log progress and enforce max steps
        class ProgressCallback(BaseCallback):
            def __init__(self, total_steps, interval):
                super().__init__()
                self.total_steps = total_steps
                self.interval = interval
                self.last_log = 0
                self.start_timesteps = 0
                
            def _on_training_start(self):
                # Record the starting timestep count
                self.start_timesteps = self.model.num_timesteps
                
            def _on_step(self):
                step = self.num_timesteps
                relative_step = step - self.start_timesteps
                
                # Log progress at specified intervals
                if step - self.last_log >= self.interval:
                    progress_pct = min(100, int(100 * relative_step / self.total_steps))
                    logger.info(f"Training progress: {relative_step}/{self.total_steps} steps ({progress_pct}%)")
                    self.last_log = step
                
                # Stop training if we've reached the maximum timesteps
                if relative_step >= self.total_steps:
                    logger.info(f"Reached maximum training steps ({self.total_steps}). Stopping training.")
                    return False
                    
                return True
                
        progress_callback = ProgressCallback(total_steps, progress_interval)
        
        # Train the agent
        try:
            self.agent.learn(
                total_timesteps=self.training_timesteps, 
                reset_num_timesteps=False,
                callback=progress_callback
            )
            logger.info(f"Training completed. Average reward: {mean_reward:.4f}")
        except Exception as e:
            if "Training ended" in str(e):
                logger.info("Training ended by callback")
            else:
                logger.error(f"Error during training: {str(e)}")
                return False
        
        # Update dynamic parameters based on mean performance
        self._update_dynamic_parameters(mean_reward)
        
        # Save episodes for future use before resetting
        if self.auto_save_episodes:
            self._save_episodes()
        
        # Reset collection after training
        self.collected_episodes = []
        
        # Update training counter
        self.training_step += 1
        
        return True
        
    def _update_dynamic_parameters(self, reward: float) -> None:
        """
        Update dynamic parameters based on recent performance.
        
        Args:
            reward: Most recent mean reward
        """
        # Adjust intervention threshold based on performance
        if reward > 0.3:  # Good performance
            # Be more conservative - increase threshold to intervene less often
            self.intervention_threshold = min(
                self.max_intervention_threshold,
                self.intervention_threshold + 0.05
            )
        else:  # Poor performance
            # Be more aggressive - decrease threshold to intervene more often
            self.intervention_threshold = max(
                self.min_intervention_threshold,
                self.intervention_threshold - 0.05
            )
            
        # Adjust training timesteps based on training progress
        progress = min(1.0, self.training_step / 50)
        self.training_timesteps = int(self.training_timesteps_min + 
                                     progress * (self.training_timesteps_max - self.training_timesteps_min))
        
        logger.debug(f"Updated parameters - Intervention threshold: {self.intervention_threshold:.2f}, " +
                    f"Training timesteps: {self.training_timesteps}")

    def update_cnn_epoch(self, epoch: int) -> None:
        """
        Update the current CNN training epoch to adjust RL training parameters.
        
        Args:
            epoch: Current CNN training epoch
        """
        # Adjust training steps based on CNN training progress
        if epoch < 80:
            # Conservative in early stages
            self.training_timesteps = self.training_timesteps_min
        elif epoch < 160:
            # Increase as we progress
            self.training_timesteps = int(self.training_timesteps_min * 1.5)
        elif epoch < 200:
            # More substantial training as we have more data
            self.training_timesteps = int(self.training_timesteps_min * 2)

    def save_brain(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Save the RL agent model.
        
        Args:
            filepath: Path to save the model, generated if not provided
            
        Returns:
            str: Path where the model was saved, or None if save failed
        """
        try:
            if filepath is None:
                import time
                filepath = os.path.join(self.brain_save_dir, f"rl_brain_{int(time.time())}.zip")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.agent.save(filepath)
            
            # Save additional metadata
            metadata = {
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

    def load_brain(self, filepath: str) -> bool:
        """
        Load the RL agent model.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            bool: True if load was successful, False otherwise
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
                self.training_step = metadata.get('training_step', self.training_step)
                self.intervention_threshold = metadata.get('intervention_threshold', self.intervention_threshold)
                
            logger.info(f"Loaded RL agent from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL agent: {str(e)}")
            return False

    def _load_episodes(self) -> None:
        """
        Load previously collected episodes from a file.
        """
        try:
            if not os.path.exists(self.episodes_save_path):
                logger.info(f"No previous episodes found at {self.episodes_save_path}. Starting with empty episodes collection.")
                return
                
            with open(self.episodes_save_path, 'r') as f:
                self.collected_episodes = json.load(f)
                
            logger.info(f"Loaded {len(self.collected_episodes)} episodes from {self.episodes_save_path}")
        except Exception as e:
            logger.error(f"Error loading episodes: {str(e)}")

    def _save_episodes(self) -> None:
        """
        Save collected episodes to a file.
        """
        try:
            with open(self.episodes_save_path, 'w') as f:
                json.dump(self.collected_episodes, f, indent=2)
                
            logger.info(f"Saved {len(self.collected_episodes)} episodes to {self.episodes_save_path}")
        except Exception as e:
            logger.error(f"Error saving episodes: {str(e)}")
