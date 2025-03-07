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
    def __init__(self, total_timesteps, verbose=1):
        super(TimestepLoggingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Log the number of timesteps done/total
        timesteps_done = self.num_timesteps
        logger.info(f"Timesteps done: {timesteps_done}, Total timesteps: {self.total_timesteps}")
        return timesteps_done < self.total_timesteps

class LRScheduler:
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
        self.intervention_threshold = config.get('intervention_threshold', 0.6)
        self.recent_intervention_scores = deque(maxlen=20)
        
        # Performance tracking for dynamic training schedule
        self.performance_history = deque(maxlen=10)
        self.reward_history = deque(maxlen=100)  # For normalizing rewards
        self.best_reward = float('-inf')  # Track best reward for saving best model

        # Track validation loss stability
        self.val_loss_history = deque(maxlen=10)
        
        # Adaptive training timesteps based on CNN training progress
        self.training_timesteps_min = config.get('training_timesteps_min', 500)
        self.training_timesteps_max = config.get('training_timesteps_max', 10000)
        self.training_timesteps = self.training_timesteps_min
        self.training_step = 0
        self.cnn_epochs_completed = 0

        # Batch learning parameters
        self.collected_episodes = []
        self.max_episodes_memory = config.get('max_episodes_memory', 20)
        self.min_episodes_for_training = config.get('min_episodes_for_training', 3)
        
        # Intervention threshold boundaries
        self.min_intervention_threshold = config.get('min_intervention_threshold', 0.3)
        self.max_intervention_threshold = config.get('max_intervention_threshold', 0.8)
        
        # Ensure brain save directory exists
        os.makedirs(self.brain_save_dir, exist_ok=True)

        # Initialize PPO agent with learning rate scheduler
        logger.info(f"Initializing PPO agent with learning rate scheduler")
        self.lr_scheduler = LRScheduler(self.learning_rate)
        self.agent = PPO(
            "MultiInputPolicy",  # Changed from "MlpPolicy" to "MultiInputPolicy"
            env,
            learning_rate=self.lr_scheduler,
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

        self.intervention_frequency = config.get('intervention_frequency', 10)
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
        if len(self.performance_history) >= 3:
            recent_rewards = list(self.performance_history)
            
            # Check if rewards are improving or declining
            is_improving = recent_rewards[-1] > np.mean(recent_rewards[:-1])
            
            # Adjust training frequency based on performance
            if is_improving:
                # If improving, we can train less frequently (save compute)
                self.train_frequency = min(self.train_frequency_max, self.train_frequency + 2)
            else:
                # If not improving, train more frequently
                self.train_frequency = max(self.train_frequency_min, self.train_frequency - 3)
            
            # Adjust the intervention threshold based on performance
            # If model is performing well (high rewards), be more conservative with interventions
            # If model is performing poorly, be more aggressive with interventions
            if np.mean(recent_rewards) > 0.3:  # Good performance
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
            
            logger.debug(f"Dynamic parameters updated - Train frequency: {self.train_frequency}, " +
                        f"Timesteps: {self.training_timesteps}, " +
                        f"Intervention threshold: {self.intervention_threshold:.2f}")

    def optimize_hyperparameters(self, observation: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Decide whether to intervene based on performance trends and agent's policy.
        Takes into account both immediate metrics and longer-term trends.
        """
        self.last_observation = observation.copy()

        try:
            # Ensure observation matches expected shape
            if observation['metrics'].shape != self.env.observation_space['metrics'].shape or \
               observation['hyperparams'].shape != self.env.observation_space['hyperparams'].shape:
                # If observation includes trend data but environment doesn't support it yet,
                # truncate to fit the expected shape
                observation['metrics'] = np.resize(observation['metrics'], self.env.observation_space['metrics'].shape)
                observation['hyperparams'] = np.resize(observation['hyperparams'], self.env.observation_space['hyperparams'].shape)

            # Get action from PPO agent
            action, _states = self.agent.predict(observation)

            self.last_action = action[0].copy()
            
            # Extract intervention score (first action component)
            intervention_score = action[0][0] if isinstance(action[0], np.ndarray) else action[0]
            
            # Track past intervention scores
            self.recent_intervention_scores.append(intervention_score)

            # Log detailed score and threshold
            logger.debug(f"Intervention score: {intervention_score:.4f} (threshold: {self.intervention_threshold:.4f})")
            logger.debug(f"Observation values: val_acc={observation['metrics'][0]:.4f}, val_loss_norm={observation['metrics'][1]:.4f}")
            
            # If trend data is included (extended observation), log that too
            if len(observation['metrics']) > 18:
                logger.debug(f"Trend data: improvement_rate={observation['metrics'][14]:.4f}, "
                           f"loss_trend={observation['metrics'][15]:.4f}, acc_trend={observation['metrics'][16]:.4f}, "
                           f"is_stagnating={observation['metrics'][17]:.1f}")

            # If score below threshold, don't intervene
            if intervention_score < self.intervention_threshold:
                return None

            # Convert action to hyperparameters with improved error handling
            try:
                if isinstance(action[0], np.ndarray):
                    action = action[0].tolist()
                hyperparams = self.env.action_to_hp_dict(action)
                # Validate hyperparameters
                required_keys = ['learning_rate', 'dropout_rate', 'weight_decay', 'optimizer_type']
                if not all(key in hyperparams for key in required_keys):
                    logger.warning(f"Missing required keys in hyperparams: {hyperparams}")
                    # Provide defaults for missing keys
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
                return None
            
            # Log the intervention decision with proposed hyperparameters
            logger.info(f"RL agent decided to intervene with hyperparams: {hyperparams}")
            return hyperparams

        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            return None

    def learn_from_episode(self, episode_rewards):
        """
        Collect episode rewards without immediately training the agent.
        Training only happens after collecting enough episodes.
        
        Args:
            episode_rewards (List[float]): List of rewards from the episode
        """
        try:
            if not episode_rewards:
                logger.warning("Empty episode rewards, skipping collection")
                return False
                
            # Store this episode's rewards
            self.collected_episodes.append(episode_rewards)
            logger.info(f"Collected episode with {len(episode_rewards)} reward(s). " + 
                       f"Total episodes: {len(self.collected_episodes)}/{self.min_episodes_for_training}")
            
            # Keep memory limited
            if len(self.collected_episodes) > self.max_episodes_memory:
                self.collected_episodes = self.collected_episodes[-self.max_episodes_memory:]
            
            # Only train if we have enough episodes
            if len(self.collected_episodes) < self.min_episodes_for_training:
                logger.info(f"Not enough episodes for training yet. " +
                           f"Have {len(self.collected_episodes)}, need {self.min_episodes_for_training}")
                return False
                
            # When we have enough episodes, trigger the actual training
            return self._train_agent_with_collected_episodes()

        except Exception as e:
            logger.error(f"Error during RL agent episode collection: {str(e)}")
            return False
            
    def _train_agent_with_collected_episodes(self): # TODO: is it even using previous rewards or past interventions
        """
        Train the agent using all collected episodes.
        This separates collection from actual training.
        """
        # Extract all rewards from collected episodes
        all_rewards = []
        for ep in self.collected_episodes:
            all_rewards.extend(ep)
                
        if not all_rewards:
            logger.warning("No rewards to train on despite having episodes")
            return False
            
        # Normalize rewards across all episodes
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        std_reward = max(std_reward, 0.1)  # Prevent division by zero
        normalized_rewards = [(r - mean_reward) / std_reward for r in all_rewards]
        normalized_rewards = [max(min(r, 5.0), -5.0) for r in normalized_rewards]
        
        logger.info(f"Training PPO agent with {len(normalized_rewards)} rewards from {len(self.collected_episodes)} episodes")
        logger.info(f"Average reward: {mean_reward:.4f}, training for {self.training_timesteps} timesteps")
        
        # Train the agent with collected experiences
        timestep_logging_callback = TimestepLoggingCallback(total_timesteps=self.training_timesteps)
        self.agent.learn(
            total_timesteps=self.training_timesteps, 
            reset_num_timesteps=False,
            callback=timestep_logging_callback
        )
        
        # Update dynamic parameters based on mean performance
        self._update_dynamic_parameters(mean_reward)
        
        # Reset collection after training
        self.collected_episodes = []
        
        # Update training counter
        self.training_step += 1
        progress = min(1.0, self.training_step / 1000)
        current_lr = self.learning_rate * np.exp(-progress)
        logger.info(f"Completed RL training. Current LR: {current_lr:.6f}, Progress: {progress:.2f}")
        
        return True

    def update_cnn_epoch(self, epoch):
        """
        Update the current CNN training epoch to adjust RL training parameters.
        
        Args:
            epoch (int): Current CNN training epoch
        """
        self.cnn_epochs_completed = epoch
        
        # Adjust training steps based on CNN training progress
        if epoch < 30:
            # Very conservative in early stages
            self.training_timesteps = self.training_timesteps_min
        elif epoch < 50:
            # Gradually increase
            self.training_timesteps = int(self.training_timesteps_min * 1.5)
        elif epoch < 75:
            # More substantial training as we have more data
            self.training_timesteps = int(self.training_timesteps_min * 2)

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
