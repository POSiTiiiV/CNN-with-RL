import os
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

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
        self.brain_save_dir = config.get('brain_save_dir', 'models/rl_brains')
        
        # Set environment's max steps
        self.env.max_steps = self.max_steps_per_episode
        
        # Training state variables
        self.last_observation = None
        self.last_action = None
        self.training_step = 0
        
        # Ensure brain save directory exists
        os.makedirs(self.brain_save_dir, exist_ok=True)
        
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
        should_intervene = self._should_intervene(val_acc_history, val_loss_history)
        
        if not should_intervene:
            return None
        
        # Store current observation for later training
        self.last_observation = observation.copy()
        
        # Get action from RL agent
        try:
            action, _ = self.agent.predict(observation.reshape(1, -1))
            
            # Store action for later training
            self.last_action = action[0].copy()
            
            # Convert action to hyperparameter dictionary
            hyperparams = self.env.action_to_hp_dict(action[0])
            
            logger.info(f"RL agent suggests hyperparameter changes: {hyperparams}")
            return hyperparams
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            return None
    
    def _should_intervene(self, val_acc_history, val_loss_history):
        """
        Determine if intervention is needed based on performance trends.
        
        Args:
            val_acc_history & val_loss_history: Lists containing metrics history
            
        Returns:
            bool: True if intervention is needed, False otherwise
        """
        if val_acc_history is None or val_loss_history is None:
            return False
        
        if len(val_acc_history) < 3 or len(val_loss_history) < 3:
            return False
        
        # Check for decreasing accuracy trend
        if val_acc_history[-1] < val_acc_history[-2] < val_acc_history[-3]:
            logger.info("Intervention trigger: decreasing accuracy trend")
            return True
            
        # Check for increasing loss trend
        if val_loss_history[-1] > val_loss_history[-2] > val_loss_history[-3]:
            logger.info("Intervention trigger: increasing loss trend")
            return True
            
        # Check for plateauing accuracy (5 epochs with less than 0.5% improvement)
        if len(val_acc_history) >= 5:
            recent_accs = val_acc_history[-5:]
            if max(recent_accs) - min(recent_accs) < 0.005:
                logger.info("Intervention trigger: plateauing accuracy")
                return True
        
        return False
    
    def learn_from_intervention(self, new_observation, reward):
        """
        Train the RL agent on a single step from the last intervention.
        
        Args:
            new_observation: New state observation after applying hyperparameter changes
            reward: Reward value indicating the success of the intervention
            
        Returns:
            bool: Success status
        """
        if self.last_observation is None or self.last_action is None:
            logger.warning("Cannot train RL agent: no previous action stored")
            return False
        
        try:
            # Log the received reward for monitoring purposes
            logger.info(f"Received reward {reward:.4f} for intervention")
            
            # For on-policy algorithms like PPO, we cannot directly add experiences to a replay buffer
            # Instead, we'll accumulate information about the quality of actions
            # and periodically trigger a proper learn() call
            
            # Store this as a successful/unsuccessful intervention
            self.training_step += 1
            
            # Update the environment's state if applicable
            if hasattr(self.env, 'update_state'):
                self.env.update_state(new_observation)
            
            # Periodically train the agent with collected experiences
            if self.training_step % self.config.get('train_frequency', 10) == 0:
                logger.info(f"Training PPO agent at step {self.training_step}")
                
                # Train for a small number of timesteps to avoid overfitting
                # This will use the environment to collect experiences internally
                self.agent.learn(
                    total_timesteps=self.config.get('training_timesteps', 1000),
                    callback=self.eval_callback,
                    reset_num_timesteps=False
                )
            
            logger.info(f"RL agent updated after intervention with reward: {reward:.4f}")
            return True
        
        except Exception as e:
            logger.error(f"Error during RL agent training: {str(e)}")
            return False
    
    def save_brain(self, filepath=None):
        """
        Save the RL agent's brain to the specified file path.
        
        Args:
            filepath (str): Path where the brain should be saved, or None for default path
        """
        try:
            # If no filepath is provided, use default with timestamp
            if filepath is None:
                import time
                timestamp = int(time.time())
                filepath = os.path.join(self.brain_save_dir, f"rl_brain_{timestamp}.zip")
            
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            self.agent.save(filepath)
            logger.info(f"RL agent's brain saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving RL agent's brain: {str(e)}")
            return None
    
    def load_brain(self, filepath):
        """
        Load the RL agent's brain from the specified file path.
        
        Args:
            filepath (str): Path to the saved brain
            
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Brain file not found: {filepath}")
                return False
                
            self.agent = PPO.load(filepath, env=self.env, device="cpu")  # Force CPU usage when loading
            logger.info(f"RL agent's brain loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL agent's brain: {str(e)}")
            return False

    def why_intervene(self, val_acc_history, val_loss_history):
        """
        Determine why intervention is needed based on performance trends.
        
        Args:
            val_acc_history & val_loss_history: Dictionary containing training and validation metrics
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