import os
import time
import json
import signal
import logging
import numpy as np
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import copy
import wandb
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from io import StringIO
from ..utils.utils import create_observation, enhance_observation_with_trends, calculate_performance_trends, is_significant_hyperparameter_change

# Set up logger
logger = logging.getLogger("cnn_rl.trainer")
# Initialize rich console for formatted output
console = Console()

# Set up file logger for tables
file_logger = logging.getLogger("file_logger")
file_handler = logging.FileHandler("training.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(message)s"))  # Plain text format
file_logger.addHandler(file_handler)
file_logger.propagate = False  # Prevent double logging

class ModelTrainer:
    """
    Manages CNN training with RL-based hyperparameter optimization.
    
    This class coordinates the training of a CNN model while using
    a reinforcement learning agent to optimize hyperparameters when
    performance plateaus.
    """
    
    def __init__(self, cnn_trainer, rl_optimizer, config):
        """
        Initialize the model trainer.
        
        Args:
            cnn_trainer: CNN trainer instance
            rl_optimizer: RL-based hyperparameter optimizer
            config: Configuration dictionary
        """
        self.cnn_trainer = cnn_trainer
        self.rl_optimizer = rl_optimizer
        self.config_training = config['training']
        self.config_logging = config['logging']
        self.config_rl = config['rl']
        
        # Get training parameters
        self.max_epochs = self.config_training.get('max_epochs', 100)
        self.early_stopping_patience = self.config_training.get('early_stopping_patience', 25)
        self.checkpoint_frequency = self.config_training.get('checkpoint_frequency', 5)
        self.min_epochs_before_intervention = self.config_training.get('min_epochs_before_intervention', 10)
        
        # RL intervention parameters
        self.metric_window_size = self.config_training.get('metric_window_size', 5)  # Rolling window size
        self.improvement_threshold = self.config_training.get('improvement_threshold', 0.002)  # Minimum expected improvement
        self.intervention_frequency = self.config_training.get('intervention_frequency', 5)
        
        # Setup logging
        self.log_dir = self.config_logging.get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'hyperparameters': [],
            'rl_interventions': [],
            'time_per_epoch': [],
            'timestamp': []
        }
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.start_time = None
        self.last_intervention_epoch = 0
        self.intervention_count = 0
        self.last_intervention_metrics = None
        self.training_interrupted = False
        
        # Episode collection for RL training (SARST format)
        self.collected_experiences = []  # Current episode experiences
        self.collected_episodes = []     # Complete episodes for tracking
        
        # Initialize history file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_file = os.path.join(self.log_dir, f"training_history_{timestamp}.json")
        training_history_dir = 'training_history'
        os.makedirs(training_history_dir, exist_ok=True)
        self.fixed_history_file = os.path.join(training_history_dir, f"training_history_{timestamp}.json")
        
        # Setup signal handling for graceful termination
        self._setup_signal_handling()
        
        # Initialize wandb
        self.use_wandb = self.config_logging.get('use_wandb', True)
        self.wandb_initialized = True
        self.global_wandb_step = 0
        
        # Create RL brain directory
        self.rl_brain_dir = os.path.join(self.log_dir, 'rl_brains')
        os.makedirs(self.rl_brain_dir, exist_ok=True)

    def train(self, epochs: Optional[int] = None) -> Dict[str, List]:
        """
        Main training loop with RL interventions.
        
        Args:
            epochs: Number of epochs to train (overrides config value if provided)
            
        Returns:
            dict: Training history
        """
        if epochs is not None:
            self.max_epochs = epochs
            
        try:
            self.start_time = time.time()
            logger.info("[bold blue]Starting Training")
            logger.info(f"[bold green]Starting training for {self.max_epochs} epochs")
            
            # Initial hyperparameters
            initial_hyperparams = self._get_current_hyperparams()
            self.history['hyperparameters'].append(initial_hyperparams)
            
            # Main training loop
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                
                # Train for one epoch
                epoch_start_time = time.time()
                train_metrics = self.cnn_trainer.train_epoch()
                epoch_time = time.time() - epoch_start_time
                
                val_metrics = self.cnn_trainer.evaluate()
                
                # Update history with metrics
                self._update_history(epoch, train_metrics, val_metrics, epoch_time)
                
                # Check if validation metrics are available
                if val_metrics['loss'] is not None:
                    val_loss = val_metrics['loss']
                    val_acc = val_metrics['accuracy']
                    
                    # Check for improvement
                    improved = False
                    if val_acc > self.best_val_acc:
                        improved = True
                        self.best_val_acc = val_acc
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        
                        # Save best model
                        self._save_best_model()
                    else:
                        self.epochs_without_improvement += 1
                    
                    # Create a summary table for the epoch
                    table = Table(title=f"Epoch {epoch+1}/{self.max_epochs}")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Training", style="green")
                    table.add_column("Validation", style="yellow")
                    
                    # Add rows with metrics
                    table.add_row("Loss", f"{train_metrics['loss']:.4f}", f"{val_loss:.4f}")
                    table.add_row("Accuracy", f"{train_metrics['accuracy']:.4f}", f"{val_acc:.4f}")
                    table.add_row("Time", f"{epoch_time:.1f}s", "")
                    table.add_row("No improvement", f"{self.epochs_without_improvement}/{self.early_stopping_patience}", "")
                    
                    # Display the table in console
                    console.print(table)
                    
                    # Convert table to plain text for log file
                    buf = StringIO()
                    temp_console = Console(file=buf, force_terminal=False, highlight=False, color_system=None)
                    temp_console.print(table)
                    table_str = buf.getvalue().strip()
                    buf.close()
                    
                    # Log to file
                    file_logger.info("\n" + table_str)
                    
                    # Show improvement message if applicable
                    if improved:
                        logger.info(f"[bold green]✓ New best model saved! (val_acc: {val_acc:.4f})[/bold green]")
                    
                    # Check for early stopping
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        logger.info(f"[bold yellow]Early stopping triggered after {epoch+1} epochs[/bold yellow]")
                        break
                    
                    # Log to wandb
                    self._log_to_wandb(epoch, train_metrics, val_metrics, epoch_time)
                    
                    # Check if RL intervention is needed
                    epochs_since_intervention = epoch + 1 - self.last_intervention_epoch
                    if epochs_since_intervention >= self.intervention_frequency:
                        logger.info("[yellow]Checking if RL intervention is needed...[/yellow]")
                        if self._check_for_intervention():
                            self.last_intervention_epoch = epoch + 1
                        else:
                            logger.info(f"[green]No intervention needed at epoch {epoch+1}[/green]")
                else:
                    # For training only metrics
                    table = Table(title=f"Epoch {epoch+1}/{self.max_epochs} (Training Only)")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")
                    
                    # Add rows with metrics
                    table.add_row("Loss", f"{train_metrics['loss']:.4f}")
                    table.add_row("Accuracy", f"{train_metrics['accuracy']:.4f}")
                    table.add_row("Time", f"{epoch_time:.1f}s")
                    
                    console.print(table)
                    
                    # Convert table to plain text for log file
                    buf = StringIO()
                    temp_console = Console(file=buf, force_terminal=False, highlight=False, color_system=None)
                    temp_console.print(table)
                    table_str = buf.getvalue().strip()
                    buf.close()
                    
                    # Log to file
                    file_logger.info("\n" + table_str)
                    
                    # Log to wandb (training only)
                    self._log_to_wandb(epoch, train_metrics, None, epoch_time)
                
                # Save checkpoint if needed
                if (epoch + 1) % self.checkpoint_frequency == 0:
                    logger.info("[bold green]Saving checkpoint...")
                    self._save_checkpoint()
                
                # Save history after each epoch
                self._save_history()
            
            # Final checkpoint and cleanup
            logger.info("[bold green]Saving final checkpoint...")
            self._save_checkpoint()
            self._save_history()
            self._plot_training_history()
            
            total_time = time.time() - self.start_time
            
            # Final summary panel
            summary = Panel(
                f"""[bold]Training Summary:[/bold]
Total time: [cyan]{total_time:.1f}s[/cyan]
Best validation accuracy: [green]{self.best_val_acc:.4f}[/green]
Total epochs: [cyan]{self.current_epoch + 1}[/cyan]
RL interventions: [yellow]{len(self.history['rl_interventions'])}[/yellow]""",
                title="[bold]Training Complete[/bold]",
                border_style="green"
            )
            logger.info(summary)
            
            # Log final summary to wandb
            if self.wandb_initialized:
                wandb.log({
                    "best_val_acc": self.best_val_acc,
                    "best_val_loss": self.best_val_loss,
                    "final_epoch": self.current_epoch + 1,
                    "total_training_time": total_time,
                    "rl_interventions_count": len(self.history['rl_interventions'])
                })
                
                # Save the final history plot to wandb
                if os.path.exists(os.path.join(self.log_dir, 'training_history.png')):
                    wandb.log({"training_history_plot": wandb.Image(
                        os.path.join(self.log_dir, 'training_history.png'),
                        caption="Training History"
                    )})
            
            return self.history
            
        except Exception as e:
            logger.exception("Error during training")
            logger.error(f"Error during training: {str(e)}")
            self._save_history()  # Save history even on error
            raise
        finally:
            # Ensure we save history on any exit
            self._cleanup()

    def _log_to_wandb(self, epoch: int, train_metrics: Dict[str, float], 
                     val_metrics: Optional[Dict[str, float]], epoch_time: float) -> None:
        """
        Log metrics and hyperparameters to Weights & Biases.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict (can be None)
            epoch_time: Time taken for this epoch
        """
        if not self.wandb_initialized:
            return
            
        try:
            # Use global step counter instead of epoch to ensure monotonicity
            self.global_wandb_step += 1
            current_step = self.global_wandb_step
            
            # Prepare metrics to log
            metrics = {
                "epoch": epoch + 1,
                "train/loss": train_metrics['loss'],
                "train/accuracy": train_metrics['accuracy'],
                "time/epoch_seconds": epoch_time,
            }
            
            # Add validation metrics if available
            if val_metrics and val_metrics['loss'] is not None:
                metrics.update({
                    "val/loss": val_metrics['loss'],
                    "val/accuracy": val_metrics['accuracy'],
                })
            
            # Add current hyperparameters
            current_hyperparams = self._get_current_hyperparams()
            for param_name, param_value in current_hyperparams.items():
                # Skip non-scalar values for direct logging
                if not isinstance(param_value, (int, float, str, bool)):
                    continue
                metrics[f"hyperparams/{param_name}"] = param_value
            
            # Add epoch progress and improvement metrics
            metrics.update({
                "training/progress": (epoch + 1) / self.max_epochs,
                "training/epochs_without_improvement": self.epochs_without_improvement,
                "training/best_val_acc": self.best_val_acc,
            })
            
            # Log to wandb with the global step
            wandb.log(metrics, step=current_step)
            
        except Exception as e:
            logger.warning(f"Failed to log to Weights & Biases: {str(e)}")

    def _check_for_intervention(self) -> bool:
        """
        Check if RL intervention is needed based on recent performance trends.
        
        Returns:
            bool: True if intervention occurred, False otherwise
        """
        # Only consider intervention after minimum epochs
        if self.current_epoch < self.min_epochs_before_intervention:
            logger.info(f"[yellow]Skipping intervention check: not enough epochs ({self.current_epoch}/{self.min_epochs_before_intervention})[/yellow]")
            return False
            
        # Calculate performance trends
        trend_data = calculate_performance_trends({
            'history': self.history,
            'metric_window_size': self.metric_window_size,
            'improvement_threshold': self.improvement_threshold,
            'loss_stagnation_threshold': self.config_training.get('loss_stagnation_threshold', 0.003)
        })
        
        # Log trend data
        is_stagnating = trend_data['is_stagnating'] 
        improvement_rate = trend_data['improvement_rate']
        
        # Create a trend summary table
        trend_table = Table(title="Performance Trends")
        trend_table.add_column("Metric", style="cyan")
        trend_table.add_column("Value", style="yellow")
        trend_table.add_row("Stagnating", f"{'Yes' if is_stagnating else 'No'}")
        trend_table.add_row("Improvement Rate", f"{improvement_rate:.5f}")
        
        # Display in console
        console.print(trend_table)
        
        # Convert table to plain text for log file
        buf = StringIO()
        temp_console = Console(file=buf, force_terminal=False, highlight=False, color_system=None)
        temp_console.print(trend_table)
        table_str = buf.getvalue().strip()
        buf.close()
        
        # Log to file
        file_logger.info("\n" + table_str)
        
        # Prepare observation for RL agent
        logger.info("[bold yellow]Querying RL agent for decision...")
        current_hyperparams = self._get_current_hyperparams()
        observation = create_observation({
            'history': self.history,
            'current_hyperparams': current_hyperparams,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'current_step': self.current_epoch,
            'max_steps': self.max_epochs,
            'no_improvement_count': self.epochs_without_improvement,
            'patience': self.early_stopping_patience
        })
        observation = enhance_observation_with_trends(observation, trend_data)
        
        # Get RL agent's decision
        new_hyperparams = self.rl_optimizer.optimize_hyperparameters(observation)
        
        # No intervention if agent decides not to intervene
        if not new_hyperparams:
            # Store metrics for non-intervention case for later reward calculation
            if self.history['val_acc']:
                self.last_intervention_metrics = {
                    'val_acc': self.history['val_acc'][-1],
                    'val_loss': self.history['val_loss'][-1],
                    'intervened': False,
                    'epoch': self.current_epoch,
                    'trend_data': trend_data,
                    'observation': observation,  # Store observation for SARST
                    'action': None               # No action taken for non-intervention
                }
            return False
            
        # Apply new hyperparameters
        logger.info(f"[bold yellow]RL agent intervening at epoch {self.current_epoch+1}[/bold yellow]")
        
        # Increment intervention counter
        self.intervention_count += 1
        
        # Store pre-intervention metrics for calculating reward later
        val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
        val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else float('inf')
        
        # Store old hyperparameters for display purposes
        old_hyperparams = copy.deepcopy(current_hyperparams)
        
        # Store metrics for reward calculation, including observation and action for SARST
        self.last_intervention_metrics = {
            'val_acc': val_acc,
            'val_loss': val_loss,
            'intervened': True,
            'epoch': self.current_epoch,
            'hyperparams': new_hyperparams.copy(),
            'trend_data': trend_data,
            'observation': observation,        # Store the observation that led to this action
            'action': new_hyperparams.copy()   # Store the action taken
        }
        
        # Determine if the intervention is major or minor
        major_intervention = self._is_major_intervention(new_hyperparams)
        
        if major_intervention:
            logger.info("[bold red]Major intervention detected, restarting training from scratch[/bold red]")
            self._restart_training(new_hyperparams)
        else:
            logger.info("[bold yellow]Minor intervention detected, continuing training[/bold yellow]")
            self._apply_hyperparams(new_hyperparams)
        
        # Show hyperparameter changes
        hp_table = Table(title="Hyperparameter Changes")
        hp_table.add_column("Parameter", style="cyan")
        hp_table.add_column("Old Value", style="yellow")
        hp_table.add_column("New Value", style="green")
        
        for key, new_val in new_hyperparams.items():
            old_val = old_hyperparams.get(key, "N/A")
            if isinstance(new_val, float):
                hp_table.add_row(key, f"{old_val:.6f}" if isinstance(old_val, float) else str(old_val), f"{new_val:.6f}")
            else:
                hp_table.add_row(key, str(old_val), str(new_val))
        
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
        
        # Record the intervention in history
        intervention = {
            'epoch': self.current_epoch + 1,
            'hyperparameters': new_hyperparams.copy(),
            'old_hyperparameters': old_hyperparams,  # Store old hyperparameters too
            'val_acc_before': val_acc,
            'val_loss_before': val_loss,
            'is_major': major_intervention
        }
        self.history['rl_interventions'].append(intervention)
        self.history['hyperparameters'].append(new_hyperparams.copy())
        
        # Log to wandb if available
        if self.wandb_initialized:
            try:
                # Use global step counter
                self.global_wandb_step += 1
                current_step = self.global_wandb_step
                
                wandb.log({
                    "rl_intervention_epoch": self.current_epoch + 1,
                    "rl_intervention_type": "major" if major_intervention else "minor",
                    "val_acc_before_intervention": val_acc,
                    "intervention_count": self.intervention_count
                }, step=current_step)
                
                for param_name, new_value in new_hyperparams.items():
                    if not isinstance(new_value, (int, float, str, bool)):
                        new_value = str(new_value)
                    wandb.log({f"rl_changes/{param_name}": new_value}, step=current_step)
            except Exception as e:
                logger.error(f"[red]Error logging to Weights & Biases: {str(e)}[/red]")
        
        return True

    def _is_major_intervention(self, new_hyperparams: Dict[str, Any]) -> bool:
        """
        Determine if the intervention is major (requiring a restart) or minor.
        
        Args:
            new_hyperparams: New hyperparameters proposed by the RL agent
            
        Returns:
            bool: True if the intervention is major, False otherwise
        """
        # Get current hyperparameters to compare against
        current_hyperparams = self._get_current_hyperparams()
        
        # Use centralized utility for determining significance of changes
        is_significant, is_major, reason = is_significant_hyperparameter_change(
            new_hyperparams, current_hyperparams
        )
        
        # Log the reason for major change if applicable
        if is_major and reason:
            logger.info(f"Major intervention: {reason}")
        elif not is_major:
            logger.info("Minor intervention: Only learning rates, dropout, or weight decay changed")
            
        return is_major

    def _restart_training(self, new_hyperparams: Dict[str, Any]) -> None:
        """
        Restart training from scratch with new hyperparameters.
        
        Args:
            new_hyperparams: New hyperparameters proposed by the RL agent
        """
        self.cnn_trainer.model.update_hyperparams(new_hyperparams)
        self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Keep history but reset the key metrics
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'hyperparameters': self.history['hyperparameters'],
            'rl_interventions': self.history['rl_interventions'],
            'time_per_epoch': [],
            'timestamp': []
        }

    def _apply_hyperparams(self, new_hyperparams: Dict[str, Any]) -> None:
        """
        Apply new hyperparameters without restarting training.
        
        Args:
            new_hyperparams: New hyperparameters proposed by the RL agent
        """
        self.cnn_trainer.model.update_hyperparams(new_hyperparams)
        self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()

    def _update_history(self, epoch: int, train_metrics: Dict[str, float], 
                        val_metrics: Dict[str, float], epoch_time: float) -> None:
        """
        Update training history with latest metrics.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            epoch_time: Time taken for this epoch
        """
        train_loss = train_metrics.get('loss', 0)
        train_acc = train_metrics.get('accuracy', 0)
        val_loss = val_metrics.get('loss', 0)
        val_acc = val_metrics.get('accuracy', 0)
        
        # Update history
        self.history['epoch'].append(epoch + 1)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['time_per_epoch'].append(epoch_time)
        self.history['timestamp'].append(datetime.now().isoformat())
        
        # Check if there was a recent intervention to calculate reward for
        if self.last_intervention_metrics is not None:
            # Wait at least 5 epochs to properly evaluate the effect
            epochs_since_intervention = self.current_epoch - self.last_intervention_metrics['epoch']
            if epochs_since_intervention >= 5:
                # Calculate reward and provide feedback to RL agent
                self._provide_intervention_reward(
                    self.last_intervention_metrics['val_acc'],
                    self.last_intervention_metrics['val_loss'],
                    val_acc,
                    val_loss,
                    self.last_intervention_metrics['intervened']
                )
                # Clear the intervention metrics
                self.last_intervention_metrics = None

    def _provide_intervention_reward(self, pre_val_acc, pre_val_loss, current_val_acc, current_val_loss, intervened):
        """
        Calculate and provide reward to the RL agent based on observed performance change.
        Stores complete SARST (State-Action-Reward-State-Terminal) experience tuples.
        """
        acc_improvement = current_val_acc - pre_val_acc
        loss_improvement = pre_val_loss - current_val_loss  # Reduction in loss is improvement
        
        # Prioritize accuracy improvement as the primary reward signal
        if intervened:
            if acc_improvement > 0.01:  # Significant improvement
                reward = 1.0 + min(acc_improvement * 10, 5.0)  # Bonus for large improvements
            elif acc_improvement > 0.002:  # Minor improvement
                reward = 0.5 + acc_improvement * 5  # Smaller bonus
            elif acc_improvement >= 0:  # Tiny or no improvement
                reward = 0.2  # Neutral/slightly positive
            else:  # Degradation
                # Penalize but not too harshly to encourage exploration
                reward = max(-1.0, -2.0 * abs(acc_improvement))
        else:
            # For non-interventions
            if acc_improvement < -0.01:  # Model got worse without intervention
                # We should have intervened, negative reward
                reward = -0.5
            else:
                # Minor changes were acceptable without intervention
                reward = 0.2  # Small positive reward
        
        # Create a complete SARST experience tuple
        if 'observation' in self.last_intervention_metrics and 'action' in self.last_intervention_metrics:
            # Get current observation (next state)
            logger.info("[bold yellow]Processing reward feedback...")
            current_hyperparams = self._get_current_hyperparams()
            current_observation = create_observation({
                'history': self.history,
                'current_hyperparams': current_hyperparams,
                'best_val_acc': self.best_val_acc,
                'best_val_loss': self.best_val_loss,
                'current_step': self.current_epoch,
                'max_steps': self.max_epochs,
                'no_improvement_count': self.epochs_without_improvement,
                'patience': self.early_stopping_patience
            })
            
            # Get trend data for the current state
            trend_data = calculate_performance_trends({
                'history': self.history,
                'metric_window_size': self.metric_window_size,
                'improvement_threshold': self.improvement_threshold,
                'loss_stagnation_threshold': self.config_training.get('loss_stagnation_threshold', 0.003)
            })
            current_observation = enhance_observation_with_trends(current_observation, trend_data)
            
            # Create the complete experience tuple
            experience = {
                'state': self.last_intervention_metrics['observation'],
                'action': self.last_intervention_metrics['action'],
                'reward': reward,
                'next_state': current_observation,
                'done': len(self.collected_experiences) == 4  # Mark as done if this is the last experience in episode
            }
            
            # Add to the current episode's experiences
            self.collected_experiences.append(experience)
        else:
            # Fallback to just storing the reward if we don't have complete data
            logger.warning("[yellow]Warning: Incomplete intervention data, storing reward only[/yellow]")
            self.collected_experiences.append(reward)
        
        # Log the reward and episode status
        if len(self.collected_experiences) == 1:  # First experience in a new episode
            logger.info(f"[bold blue]Starting new RL episode ({len(self.collected_episodes) + 1})[/bold blue]")
        
        # Determine color based on reward
        reward_color = "green" if reward > 0 else "red"
        action_type = "intervention" if intervened else "non-intervention"
        logger.info(f"[bold]RL reward for {action_type}:[/bold] [{reward_color}]{reward:.4f}[/{reward_color}] "
                     f"(acc change: [cyan]{acc_improvement:.4f}[/cyan], loss change: [cyan]{loss_improvement:.4f}[/cyan])")
        
        # Check if we've completed an episode (5 interventions)
        if len(self.collected_experiences) >= 5:
            logger.info("[bold blue]RL Episode Complete")
            logger.info(f"[bold green]Episode {len(self.collected_episodes) + 1} completed with {len(self.collected_experiences)} interventions[/bold green]")
            
            # Store the completed episode
            self.collected_episodes.append(self.collected_experiences)
            
            # Train the RL agent with collected experiences
            logger.info("[bold yellow]Training RL agent with episode data...")
            training_success = self.rl_optimizer.learn_from_episode(self.collected_experiences)
            
            # Clear the collected experiences after training
            self.collected_experiences = []
            
            if training_success:
                logger.info("[green]✓[/green] RL agent trained successfully with episode data")
            else:
                logger.error("[red]✗[/red] RL agent training failed")
            return True
        
        return False

    def _get_current_hyperparams(self) -> Dict[str, Any]:
        """
        Get current hyperparameters from model.
        
        Returns:
            dict: Current hyperparameters
        """
        return self.cnn_trainer.model.hyperparams
        
    def _save_history(self) -> None:
        """Save training history to JSON files."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_history = {}
            for key, value in self.history.items():
                if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    json_history[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in value]
                else:
                    json_history[key] = value
            
            # Save to timestamped file
            with open(self.history_file, 'w') as f:
                json.dump(json_history, f, indent=2)
                
            # Also save to fixed location for easy access
            with open(self.fixed_history_file, 'w') as f:
                json.dump(json_history, f, indent=2)
                
            logger.debug(f"Training history saved to {self.history_file} and {self.fixed_history_file}")
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.cnn_trainer.model.state_dict(),
                'optimizer_state_dict': self.cnn_trainer.optimizer.state_dict(),
                'best_val_acc': self.best_val_acc,
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'epochs_without_improvement': self.epochs_without_improvement,
                'current_hyperparams': self._get_current_hyperparams()
            }
            
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_epoch_{self.current_epoch+1}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"[green]✓[/green] Checkpoint saved at epoch {self.current_epoch+1}")
        except Exception as e:
            logger.error(f"[red]✗[/red] Error saving checkpoint: {str(e)}")

    def _save_best_model(self) -> None:
        """Save the best model based on validation accuracy."""
        try:
            best_model_path = os.path.join(self.log_dir, 'best_model.pt')
            
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.cnn_trainer.model.state_dict(),
                'optimizer_state_dict': self.cnn_trainer.optimizer.state_dict(),
                'val_acc': self.best_val_acc,
                'val_loss': self.best_val_loss,
                'hyperparams': self._get_current_hyperparams()
            }, best_model_path)
            
            logger.info(f"[green]✓[/green] Best CNN model saved with val_acc={self.best_val_acc:.4f}")
        except Exception as e:
            logger.error(f"[red]✗[/red] Error saving best CNN model: {str(e)}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            
            # Load model and optimizer states
            self.cnn_trainer.model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_val_loss = checkpoint['best_val_loss']
            self.history = checkpoint['history']
            self.epochs_without_improvement = checkpoint['epochs_without_improvement']
            
            logger.info(f"[green]✓[/green] Resumed training from epoch {self.current_epoch}")
        except Exception as e:
            logger.exception("Error loading checkpoint")
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def _cleanup(self):
        """Cleanup resources before exit."""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            logger.info(f"[dim]Training session lasted {total_time:.1f}s[/dim]")
        
        # Final save of history
        logger.info("[dim]Saving final history...[/dim]")
        self._save_history()
        
        # Finish wandb run if active
        if self.wandb_initialized:
            try:
                logger.info("[dim]Finalizing Weights & Biases logging...[/dim]")
                # Log final summary if we haven't already
                wandb.log({
                    "final_epoch": self.current_epoch + 1,
                    "best_val_acc": self.best_val_acc,
                    "completed": not self.training_interrupted,
                    "early_stopped": self.epochs_without_improvement >= self.early_stopping_patience
                }, step=self.global_wandb_step)
                
                # Finish the run properly
                wandb.finish()
                logger.info("[dim]Weights & Biases run finished[/dim]")
            except Exception as e:
                logger.warning(f"[yellow]Warning: Error finalizing wandb run: {str(e)}[/yellow]")

    def _setup_signal_handling(self):
        """
        Set up signal handlers for graceful termination when Ctrl+C is pressed.
        This allows saving the model and history before exiting.
        """
        # Register the signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.debug("Signal handler registered for graceful termination (Ctrl+C)")
    
    def _signal_handler(self, sig, frame):
        """
        Handle SIGINT (Ctrl+C) signal to gracefully terminate training.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        if not self.training_interrupted:  # Prevent multiple interruptions
            logger.warning("\n[bold yellow]⚠️ Training interrupted by user (Ctrl+C)[/bold yellow]")
            logger.info("[yellow]Saving checkpoint and cleaning up...[/yellow]")
            
            # Mark as interrupted
            self.training_interrupted = True
            
            # Save checkpoint and history
            try:
                if hasattr(self, 'current_epoch'):
                    self._save_checkpoint()
                    self._save_history()
                    
                    # Save the RL brain if available
                    if hasattr(self, 'rl_optimizer') and hasattr(self.rl_optimizer, 'save_brain'):
                        brain_path = os.path.join(self.rl_brain_dir, f"interrupted_brain_{self.current_epoch+1}.zip")
                        self.rl_optimizer.save_brain(brain_path)
                        logger.info(f"[green]✓[/green] RL brain saved to {brain_path}")
            except Exception as e:
                logger.error(f"[red]Error during cleanup after interruption: {str(e)}[/red]")
            
            # Display summary info
            if hasattr(self, 'start_time') and self.start_time is not None:
                elapsed = time.time() - self.start_time
                logger.info(f"[yellow]Training ran for {elapsed:.1f}s before interruption[/yellow]")
            
            logger.info("[bold green]Training state saved. Exiting gracefully.[/bold green]")
            
            # Properly clean up wandb if it's running
            if hasattr(self, 'wandb_initialized') and self.wandb_initialized and wandb.run is not None:
                try:
                    wandb.log({"interrupted": True})
                    wandb.finish(exit_code=0)  # Use 0 for manual interruption
                except Exception as we:
                    logger.warning(f"[yellow]Error finishing wandb run: {str(we)}[/yellow]")
                    
            # Exit with a non-zero code to indicate interruption
            # But before exiting, let Python finish any pending operations
            import sys
            sys.exit(1)

    def _plot_training_history(self):
        """
        Plot training history and save to file.
        """
        try:
            if len(self.history['epoch']) < 2:
                logger.warning("[yellow]Not enough data points to plot training history[/yellow]")
                return
                
            plt.figure(figsize=(12, 8))
            
            # Plot loss
            plt.subplot(2, 1, 1)
            plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss', color='blue')
            if self.history['val_loss'] and all(x is not None for x in self.history['val_loss']):
                plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss', color='orange')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Plot accuracy
            plt.subplot(2, 1, 2)
            plt.plot(self.history['epoch'], self.history['train_acc'], label='Train Acc', color='blue')
            if self.history['val_acc'] and all(x is not None for x in self.history['val_acc']):
                plt.plot(self.history['epoch'], self.history['val_acc'], label='Val Acc', color='orange')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            
            # Mark interventions
            if self.history['rl_interventions']:
                for intervention in self.history['rl_interventions']:
                    epoch = intervention['epoch']
                    plt.subplot(2, 1, 1)
                    plt.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
                    plt.subplot(2, 1, 2)
                    plt.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(self.log_dir, 'training_history.png'), dpi=300)
            plt.close()
            
            logger.info(f"[green]✓[/green] Training history plot saved to {self.log_dir}/training_history.png")
        except Exception as e:
            logger.warning(f"[yellow]Failed to plot training history: {str(e)}[/yellow]")