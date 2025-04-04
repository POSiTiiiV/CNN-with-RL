import os
import time
import numpy as np
import logging
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from io import StringIO
from ..utils.utils import is_significant_hyperparameter_change

# Set up logger using the consistent approach as in main.py
logger = logging.getLogger("cnn_rl.rl_agent")

# Create a console for rendering tables
console = Console(highlight=False)  # Disable syntax highlighting to avoid unexpected color codes

# Set up file logger for tables
file_logger = logging.getLogger("file_logger")
file_handler = logging.FileHandler("training.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(message)s"))  # Plain text format
file_logger.addHandler(file_handler)
file_logger.propagate = False  # Prevent double logging

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
                self.model.save("best_rl_brain")
                logger.info(f"Saved new best RL brain with mean reward: {self.best_mean_reward:.2f}")
                
        return True

class ProgressCallback(BaseCallback):
    """
    Callback to log training progress and enforce maximum timesteps limit.
    This callback suppresses wandb logging during progress reporting to prevent
    flickering of the rich progress bar.
    """
    def __init__(self, total_steps, interval):
        super().__init__()
        self.total_steps = total_steps
        self.interval = interval
        self.last_log = 0
        self.start_timesteps = 0
        self.original_level = logging.getLogger().level

    def _on_training_start(self):
        """Initialize progress tracking at the start of training"""
        # Record the starting timestep count
        self.start_timesteps = self.model.num_timesteps
        logger.info(f"Starting training from timestep {self.start_timesteps} for {self.total_steps} steps")
        
    def _on_step(self):
        """Log progress after each step and check for completion"""
        step = self.num_timesteps
        relative_step = step - self.start_timesteps

        # Log progress periodically
        if relative_step % max(1, self.total_steps // 20) == 0 or relative_step >= self.total_steps:
            progress_pct = min(100, int(100 * relative_step / self.total_steps))
            logger.info(f"Steps: {relative_step}/{self.total_steps} ({progress_pct}%)")

        # Stop training if we've reached the maximum timesteps
        if relative_step >= self.total_steps:
            logger.info(f"Reached maximum training steps ({self.total_steps}). Stopping training.")
            return False

        return True
    
    def _on_training_end(self):
        """Clean up when training ends"""
        logger.info("Training completed")

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
        logger.info("[bold blue]Initializing PPO agent[/bold blue]")
        self.lr_scheduler = LRScheduler(self.learning_rate)
        
        # Create status message
        status_message = "[yellow]Creating PPO agent on CPU..."
        logger.info(status_message)
        
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
        
        # Connect agent to environment for direct saving
        if hasattr(self.env, 'set_agent'):
            self.env.set_agent(self.agent)
            logger.info("[green]✓[/green] Agent connected to environment for direct saving")

        # Store most recent action and state for learning
        self.last_state = None
        self.last_action = None

        # Load previously collected episodes if available and configured
        if self.auto_load_episodes:
            logger.info("[yellow]Loading previous episodes...")
            self._load_episodes()
            
            # Train the agent with loaded episodes immediately if we have enough
            if len(self.collected_episodes) >= self.min_episodes_for_training:
                logger.info(f"[bold green]Found {len(self.collected_episodes)} previously collected episodes.[/bold green] Training RL agent immediately.")
                self._train_agent_with_collected_episodes()
            else:
                logger.info(f"[yellow]Found {len(self.collected_episodes)} previously collected episodes.[/yellow] Need at least {self.min_episodes_for_training} for training.")

        logger.info("[bold green]✓[/bold green] HyperParameterOptimizer initialized on CPU")

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
                    logger.warning("[yellow]Warning:[/yellow] Found NaN values in observation key '{key}', replacing with zeros")
                    observation[key] = np.nan_to_num(value, nan=0.0)

            # Get action from PPO agent (intervention decision + hyperparameter changes)
            try:
                logger.info("[yellow]Querying RL agent for decision...")
                action, _states = self.agent.predict(observation)
            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    logger.error(f"[red]Numerical instability in prediction: {e}. Returning no intervention.[/red]")
                    return None
                raise e
            
            # Check for NaN values in the action
            if isinstance(action, np.ndarray) and np.isnan(action).any():
                logger.error("[red]Action contains NaN values, returning no intervention[/red]")
                return None
                
            # Extract intervention score (first action component)
            intervention_score = action[0][0] if isinstance(action[0], np.ndarray) else action[0]
            
            # Additional check for NaN in intervention score
            if np.isnan(intervention_score):
                logger.error("[red]Intervention score is NaN, returning no intervention[/red]")
                return None
                
            # Track recent scores
            self.recent_intervention_scores.append(intervention_score)
            
            # Create a colored indicator for intervention score
            score_color = "green" if intervention_score >= self.intervention_threshold else "yellow"
            logger.info(f"Intervention score: [{score_color}]{intervention_score:.4f}[/{score_color}] (threshold: {self.intervention_threshold:.4f})")
            
            # If score below threshold, don't intervene
            if intervention_score < self.intervention_threshold:
                logger.info(f"[yellow]⚠[/yellow] Intervention score below threshold, no intervention needed")
                return None 

            # Convert action to hyperparameters
            try:
                if isinstance(action[0], np.ndarray):
                    action = action[0].tolist()
                
                # Check for NaN values in action before conversion
                if any(np.isnan(x) if isinstance(x, (float, np.float32, np.float64)) else False for x in action):
                    logger.error("[red]NaN values in action, returning no intervention[/red]")
                    return None
                
                hyperparams = self.env.action_to_hp_dict(action)
                
                # Verify hyperparams don't contain NaN values
                for k, v in hyperparams.items():
                    if isinstance(v, (float, np.float32, np.float64)) and np.isnan(v):
                        logger.error(f"[red]NaN value in hyperparameter {k}, returning no intervention[/red]")
                        return None
            except Exception as e:
                logger.exception("Error converting action to hyperparameters")
                return None

            # Check for identical hyperparameters (exact equality check)
            if current_hyperparams is not None and all(
                hyperparams.get(key) == current_hyperparams.get(key)
                for key in set(hyperparams.keys()) | set(current_hyperparams.keys())
            ):
                logger.warning("[yellow]Proposed hyperparameters are identical to current ones, skipping intervention[/yellow]")
                return None

            # Display proposed hyperparameters in a table
            hp_table = Table(title="Proposed Hyperparameters")
            hp_table.add_column("Parameter", style="cyan")
            hp_table.add_column("Value", style="green")
            
            for param_name, param_value in hyperparams.items():
                if isinstance(param_value, float):
                    hp_table.add_row(param_name, f"{param_value:.6f}")
                else:
                    hp_table.add_row(param_name, str(param_value))
            
            # Display in console
            console.print(hp_table)
            
            # Convert table to plain text for log file
            buf = StringIO()
            temp_console = Console(file=buf, force_terminal=False, highlight=False, color_system=None)
            temp_console.print(hp_table)
            table_str = buf.getvalue().strip()
            buf.close()
            
            # Log to file
            file_logger.info("\n" + table_str)

            # Use centralized utility to check if hyperparameter changes are significant
            if current_hyperparams is not None:
                is_significant, is_major, reason = is_significant_hyperparameter_change(
                    hyperparams, current_hyperparams
                )
                
                if not is_significant:
                    logger.warning("[yellow]Proposed hyperparameters too similar to current ones, skipping intervention[/yellow]")
                    return None
                
                if is_major:
                    logger.info(f"[bold red]Major hyperparameter change detected:[/bold red] {reason}")
                else:
                    logger.info(f"[bold green]Significant hyperparameter changes detected, proceeding with intervention[/bold green]")
    
            return hyperparams

        except Exception as e:
            logger.exception("Error during hyperparameter optimization")
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
                logger.warning("[yellow]Warning:[/yellow] Empty episode experiences, skipping collection")
                return False
                
            # Store this episode's experiences
            self.collected_episodes.append(episode_experiences)
            logger.info(f"Collected episode with [cyan]{len(episode_experiences)}[/cyan] experiences. " + 
                         f"Total episodes: [cyan]{len(self.collected_episodes)}/{self.min_episodes_for_training}[/cyan]")
            
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
            logger.exception(f"Error during RL agent episode collection: {str(e)}")
            return False
        
    def _train_agent_with_collected_episodes(self) -> bool:
        """
        Train the agent using all collected episodes with SARST format.
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        if not self.collected_episodes:
            logger.warning("[yellow]Warning:[/yellow] No episodes to train on")
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
            logger.warning("[yellow]Warning:[/yellow] No rewards found in collected episodes")
            return False
            
        # Calculate training statistics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        min_reward = min(all_rewards)
        max_reward = max(all_rewards)
        
        # Create summary table
        stats_table = Table(title="Training Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Experiences", str(experience_count))
        stats_table.add_row("Episodes", str(len(self.collected_episodes)))
        stats_table.add_row("Mean reward", f"{mean_reward:.4f}")
        stats_table.add_row("Std dev", f"{std_reward:.4f}")
        stats_table.add_row("Min reward", f"{min_reward:.4f}")
        stats_table.add_row("Max reward", f"{max_reward:.4f}")
        
        # Display in console
        console.print(stats_table)
        
        # Convert table to plain text for log file
        buf = StringIO()
        temp_console = Console(file=buf, force_terminal=False, highlight=False, color_system=None)
        temp_console.print(stats_table)
        table_str = buf.getvalue().strip()
        buf.close()
        
        # Log to file
        file_logger.info("\n" + table_str)
        
        logger.info("[bold blue]Starting PPO Training")
        
        # Create a progress bar for training
        total_steps = self.training_timesteps
        progress_interval = max(1, total_steps // 40)  # Report progress ~40 times
        logger.info(f"Training for [cyan]{total_steps}[/cyan] timesteps...")
        
        # Create a callback for training progress
        progress_callback = ProgressCallback(total_steps, progress_interval)
        
        # Ensure any existing live displays are cleared
        if hasattr(console, '_live') and console._live is not None:
            console.clear_live()
        
        # Train the agent - don't use console.status here as it creates a live display
        # that would conflict with the progress callback's own live display
        try:
            self.agent.learn(
                total_timesteps=self.training_timesteps, 
                reset_num_timesteps=False,
                callback=progress_callback
            )
            logger.info(f"[green]✓[/green] Training completed. Average reward: [cyan]{mean_reward:.4f}[/cyan]")
        except Exception as e:
            if "Training ended" in str(e):
                logger.warning("[yellow]Training ended by callback[/yellow]")
            else:
                logger.exception("Error during training")
                logger.error(f"[red]Error during training: {str(e)}[/red]")
                return False
        
        # Update dynamic parameters based on mean performance
        self._update_dynamic_parameters(mean_reward)
        
        # Save episodes for future use before resetting
        if self.auto_save_episodes:
            logger.info("[yellow]Saving episodes...")
            self._save_episodes()
        
        # Save the brain after training with timestamp
        timestamp = int(time.time())
        brain_path = os.path.join(self.brain_save_dir, f"rl_brain_training_{timestamp}.zip")
        logger.info(f"[yellow]Saving RL brain to {brain_path}...")
        self.save_brain(brain_path)
        
        # If this is the best reward we've seen, also save as best_rl_brain
        if not hasattr(self, 'best_reward') or mean_reward > self.best_reward:
            self.best_reward = mean_reward
            best_brain_path = os.path.join(self.brain_save_dir, "best_rl_brain.zip")
            
            logger.info("[bold green]Saving best brain...")
            # Create metadata file with reward info
            metadata = {
                "best_reward": mean_reward,
                "training_step": self.training_step,
                "intervention_threshold": self.intervention_threshold
            }
            
            # Save metadata to a separate file
            metadata_path = os.path.join(self.brain_save_dir, "best_rl_brain_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Copy the current brain to best_brain location
            import shutil
            shutil.copy(brain_path, best_brain_path)
            
            logger.info(f"[bold green]✓[/bold green] New best RL brain saved with reward: [cyan]{mean_reward:.4f}[/cyan]")
        
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
        
        logger.info(f"[dim]Updated parameters - Intervention threshold: [cyan]{self.intervention_threshold:.2f}[/cyan], " +
                    f"Training timesteps: [cyan]{self.training_timesteps}[/cyan][/dim]")

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
                
            logger.info(f"[green]✓[/green] Saved RL agent brain to {filepath}")
            return filepath
        except Exception as e:
            logger.exception(f"Error saving RL agent brain: {str(e)}")
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
                logger.error(f"[red]RL brain file not found: {filepath}[/red]")
                return False
                
            logger.info(f"[yellow]Loading RL brain from {filepath}...")
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
                
            logger.info(f"[green]✓[/green] Loaded RL agent brain from {filepath}")
            return True
        except Exception as e:
            logger.exception(f"Error loading RL agent brain: {str(e)}")
            return False

    def _load_episodes(self) -> None:
        """
        Load previously collected episodes from a file.
        Uses the structured versioned format.
        """
        try:
            if not os.path.exists(self.episodes_save_path):
                logger.warning(f"[yellow]No previous episodes found at {self.episodes_save_path}. Starting with empty episodes collection.[/yellow]")
                return
                
            with open(self.episodes_save_path, 'r') as f:
                loaded_data = json.load(f)
                
            # Check if this is a valid versioned format
            if not isinstance(loaded_data, dict) or "version" not in loaded_data or "episodes" not in loaded_data:
                logger.warning(f"[yellow]Warning:[/yellow] Episodes file {self.episodes_save_path} has invalid format. Starting with empty episodes.")
                self.collected_episodes = []
                return
                
            # Extract episodes and metadata
            self.collected_episodes = loaded_data["episodes"]
            
            # Restore relevant metadata if available
            if "metadata" in loaded_data:
                metadata = loaded_data["metadata"]
                self.intervention_threshold = metadata.get("intervention_threshold", self.intervention_threshold)
                self.training_timesteps = metadata.get("training_timesteps", self.training_timesteps)
                
            logger.info(f"Loaded [cyan]{len(self.collected_episodes)}[/cyan] episodes from {self.episodes_save_path} (format version {loaded_data.get('version', 1)})")
            
            # Enforce memory limit
            if len(self.collected_episodes) > self.max_episodes_memory:
                logger.info(f"[yellow]Limiting loaded episodes[/yellow] to the most recent {self.max_episodes_memory} (from {len(self.collected_episodes)} total)")
                self.collected_episodes = self.collected_episodes[-self.max_episodes_memory:]
                
        except Exception as e:
            logger.exception(f"Error loading episodes: {str(e)}")
            # Start with empty episodes on error
            self.collected_episodes = []

    def _save_episodes(self) -> None:
        """
        Save collected RL training episodes to a file.
        Uses a versioned approach to maintain multiple sets of episodes.
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_episodes = []
            
            for episode in self.collected_episodes:
                serializable_experiences = []
                
                for exp in episode:
                    # Create a copy of the experience that we can modify
                    serializable_exp = {}
                    
                    # Convert each field in the experience
                    for key, value in exp.items():
                        # Convert numpy arrays to lists
                        if isinstance(value, np.ndarray):
                            serializable_exp[key] = value.tolist()
                        # Handle nested dictionaries (like state and next_state)
                        elif isinstance(value, dict):
                            serializable_exp[key] = {}
                            for k, v in value.items():
                                if isinstance(v, np.ndarray):
                                    serializable_exp[key][k] = v.tolist()
                                else:
                                    serializable_exp[key][k] = v
                        # Keep other types as is
                        else:
                            serializable_exp[key] = value
                            
                    serializable_experiences.append(serializable_exp)
                
                serializable_episodes.append(serializable_experiences)
            
            # Create a structured episodes file that supports versioning and metadata
            episodes_data = {
                "version": 1,
                "timestamp": time.time(),
                "training_step": self.training_step,
                "episode_count": len(serializable_episodes),
                "metadata": {
                    "intervention_threshold": self.intervention_threshold,
                    "training_timesteps": self.training_timesteps
                },
                "episodes": serializable_episodes
            }
            
            # Create episodes directory if it doesn't exist
            episodes_dir = os.path.dirname(self.episodes_save_path)
            os.makedirs(episodes_dir, exist_ok=True)
            
            # Save the current episodes to the standard path for loading next time
            with open(self.episodes_save_path, 'w') as f:
                json.dump(episodes_data, f, indent=2)
            
            # Also save a timestamped version to maintain history
            # Only keep a fixed number of history files
            self._maintain_episode_history(episodes_data)
                
            logger.info(f"Saved {len(self.collected_episodes)} RL training episodes to {self.episodes_save_path}")
        except Exception as e:
            logger.error(f"[red]Error saving RL training episodes: {str(e)}[/red]")
            
    def _maintain_episode_history(self, episodes_data):
        """
        Maintain a history of episode files by saving timestamped versions
        and cleaning up old files when there are too many.
        
        Args:
            episodes_data: The episode data to save
        """
        try:
            # Define max number of history files to keep
            max_history_files = 10
            
            # Create a timestamped filename
            timestamp = int(time.time())
            base_dir = os.path.dirname(self.episodes_save_path)
            base_filename = os.path.basename(self.episodes_save_path)
            history_dir = os.path.join(base_dir, "episodes_history")
            os.makedirs(history_dir, exist_ok=True)
            
            # Name format: original_name_timestamp.json
            history_filename = os.path.splitext(base_filename)[0] + f"_{timestamp}.json"
            history_path = os.path.join(history_dir, history_filename)
            
            # Save the timestamped version
            with open(history_path, 'w') as f:
                json.dump(episodes_data, f, indent=2)
                
            # List all history files
            history_files = []
            for f in os.listdir(history_dir):
                if f.endswith('.json') and f.startswith(os.path.splitext(base_filename)[0]):
                    file_path = os.path.join(history_dir, f)
                    history_files.append((os.path.getmtime(file_path), file_path))
            
            # Sort by modification time (oldest first)
            history_files.sort()
            
            # Remove oldest files if we have too many
            if len(history_files) > max_history_files:
                for _, old_file in history_files[:(len(history_files) - max_history_files)]:
                    try:
                        os.remove(old_file)
                        logger.info(f"[dim]Removed old episodes history file: {old_file}[/dim]")
                    except Exception as e:
                        logger.warning(f"[yellow]Could not remove old episodes file {old_file}: {e}[/yellow]")
                        
        except Exception as e:
            logger.warning(f"[yellow]Error maintaining episode history: {e}[/yellow]")
            # Continue execution - this is not critical
