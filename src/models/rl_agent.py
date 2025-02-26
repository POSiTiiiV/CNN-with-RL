import os
import numpy as np
import torch
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

def get_device():
    """Get available device - forcing CPU for PPO"""
    return "cpu"  # Forces CPU usage for PPO as recommended

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
            # Evaluate the agent's performance
            # For now, just log the current reward
            logger.info(f"EvaluationCallback step {self.n_calls}, "
                        f"mean reward: {self.model.ep_info_buffer.mean():.2f}")
            
            # Save best model
            if self.model.ep_info_buffer.mean() > self.best_mean_reward:
                self.best_mean_reward = self.model.ep_info_buffer.mean()
                self.model.save("best_model")
                logger.info(f"Saved new best model with mean reward: {self.best_mean_reward:.2f}")
                
        return True

class HyperParameterOptimizer:
    """
    RL-based hyperparameter optimizer for CNN models.
    Uses Proximal Policy Optimization to learn optimal hyperparameter settings.
    """
    
    def __init__(self, env, config):
        """
        Initialize the optimizer with environment and configuration.
        
        Args:
            env: Hyperparameter optimization environment
            config: Configuration dictionary
        """
        self.env = env
        self.config = config
        
        # Extract configuration
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.steps_per_episode = config.get('steps_per_episode', 5)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 5)
        self.n_steps = config.get('n_steps', 1024)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.normalize_advantage = config.get('normalize_advantage', True)
        
        # Set environment's max steps
        self.env.max_steps = self.max_steps_per_episode
        
        # Initialize the RL agent - force CPU usage
        logger.info(f"Initializing PPO agent with learning rate {self.learning_rate}")
        
        self.agent = PPO(
            "MlpPolicy",  # Standard policy for MultiDiscrete action space
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
            device="cpu"  # Force CPU usage for PPO
        )
        
        # Initialize evaluation callback
        self.eval_callback = EvaluationCallback(eval_freq=1000)
        
        logger.info("HyperParameterOptimizer initialized")
    
    def train(self, total_timesteps=100000):
        """
        Train the RL agent to optimize hyperparameters.
        
        Args:
            total_timesteps: Total number of timesteps to train the agent
            
        Returns:
            trained agent
        """
        logger.info(f"Starting RL agent training for {total_timesteps} timesteps")
        
        try:
            self.agent.learn(
                total_timesteps=total_timesteps,
                callback=self.eval_callback,
                log_interval=10
            )
            
            # Save the trained model
            self.agent.save("final_model")
            logger.info("RL agent training completed and model saved")
            
        except Exception as e:
            logger.error(f"Error during RL agent training: {str(e)}")
            raise
            
        return self.agent
    
    def optimize_hyperparameters(self, observation, val_acc_history=None, val_loss_history=None):
        """
        Determine if intervention is needed and suggest hyperparameter changes.
        
        Args:
            observation: Current system state observation
            val_acc_history: History of validation accuracies
            val_loss_history: History of validation losses
            
        Returns:
            dict: Suggested hyperparameter changes
        """
        # Check if intervention is needed
        self.why_intervene(val_acc_history, val_loss_history)
        
        # Get action from RL agent
        try:
            action, _ = self.agent.predict(observation.reshape(1, -1))
            
            # Convert action to hyperparameter dictionary
            hyperparams = self.env.action_to_hp_dict(action[0])
            
            logger.info(f"RL agent suggests hyperparameter changes: {hyperparams}")
            return hyperparams
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            return None
    
    def why_intervene(self, val_acc_history, val_loss_history):
        """
        Determine why intervention is needed based on performance trends.
        
        Args:
            history: Dictionary containing training and validation metrics
        """
        
        logged = False
        logger.info("Intervening cause: ")   
        # Check for decreasing accuracy trend
        if val_acc_history[-1] < val_acc_history[-2] < val_acc_history[-3]:
            logger.info("\t-decreasing accuracy trend")
            logged = True
        # Check for increasing loss trend
        if val_loss_history[-1] > val_loss_history[-2] > val_loss_history[-3]:
            logger.info("\t-increasing loss trend")
            logged = True
        # Check for plateauing accuracy (5 epochs with less than 0.5% improvement)
        if len(val_acc_history) >= 5:
            recent_accs = val_acc_history[-5:]
            if max(recent_accs) - min(recent_accs) < 0.005:
                logger.info("\t-plateauing accuracy")
                logged = True
        if not logged:
            logger.info("-we have to")
    
class OfflineHPOptimizer:
    """
    Offline hyperparameter optimizer that pre-trains an RL agent.
    """
    
    def __init__(self, env, config):
        """
        Initialize the optimizer with environment and configuration.
        
        Args:
            env: Hyperparameter optimization environment
            config: Configuration dictionary
        """
        self.env = env
        self.config = config
        self.optimizer = HyperParameterOptimizer(env, config)
        
    def train_offline(self, episodes=50):
        """
        Train the RL agent offline with simulated data.
        
        Args:
            episodes: Number of episodes to train
            
        Returns:
            dict: Training results summary
        """
        logger.info(f"Starting offline training for {episodes} episodes")
        
        total_timesteps = episodes * self.config.get('max_steps_per_episode', 5)
        self.optimizer.train(total_timesteps=total_timesteps)
        
        # Test the trained agent
        observation, _ = self.env.reset()
        action, _ = self.optimizer.agent.predict(observation.reshape(1, -1))
        hyperparams = self.env.action_to_hp_dict(action[0])
        
        logger.info(f"Offline training completed. Sample hyperparameters: {hyperparams}")
        
        return {
            'episodes': episodes,
            'total_timesteps': total_timesteps,
            'best_reward': self.optimizer.eval_callback.best_mean_reward,
            'final_hyperparams': hyperparams
        }
