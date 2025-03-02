import os
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

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
        self.intervention_threshold = config.get('intervention_threshold', 0.5)
        self.recent_intervention_scores = deque(maxlen=20)  # Track last 20 intervention scores

        # Training control
        self.train_frequency = config.get('train_frequency', 17)
        self.training_timesteps = config.get('training_timesteps', 1000)
        self.training_step = 0

        # Ensure brain save directory exists
        os.makedirs(self.brain_save_dir, exist_ok=True)

        # Initialize PPO agent (forcing CPU usage)
        logger.info(f"Initializing PPO agent with learning rate {self.learning_rate}")

        self.agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            normalize_advantage=self.normalize_advantage,
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
            device="cpu"
        )

        self.eval_callback = EvaluationCallback(eval_freq=1000)
        logger.info("HyperParameterOptimizer initialized")

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

            # Dynamic threshold: use moving average + small buffer
            threshold = np.mean(self.recent_intervention_scores) + 0.1
            
            logger.debug(f"Intervention score: {intervention_score:.4f} (threshold: {threshold:.4f})")

            if intervention_score < threshold:
                logger.info("RL agent decided not to intervene")
                return None

            hyperparams = self.env.action_to_hp_dict(action[0])
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

            logger.info(f"Reward assigned: {reward:.4f}")

            # Increment training step
            self.training_step += 1

            # Update environment state if needed
            if hasattr(self.env, 'update_state'):
                self.env.update_state(new_observation)

            # Adaptive training schedule
            if self.training_step % self.train_frequency == 0:
                timesteps = min(10000, max(1000, self.training_step * 100))
                logger.info(f"Training PPO agent at step {self.training_step} for {timesteps} timesteps")
                
                # Create and use the custom callback
                timestep_logging_callback = TimestepLoggingCallback(total_timesteps=timesteps)
                self.agent.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=timestep_logging_callback)

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
            logger.info(f"Saved RL agent to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving RL agent: {str(e)}")
            return None

    def load_brain(self, filepath):
        """
        Load the RL agent.
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Brain file not found: {filepath}")
                return False
            self.agent = PPO.load(filepath, env=self.env, device="cpu")
            logger.info(f"Loaded RL agent from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL agent: {str(e)}")
            return False
