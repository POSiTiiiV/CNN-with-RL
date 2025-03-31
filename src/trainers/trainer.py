import os
import time
import json
import signal
import logging
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import wandb
from ..utils.utils import create_observation, enhance_observation_with_trends, calculate_performance_trends

logger = logging.getLogger(__name__)

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
            logger.info(f"Starting training for {self.max_epochs} epochs")
            
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
                    
                    # Log progress with better formatting
                    logger.info(f"\n=== Epoch {epoch+1}/{self.max_epochs} ===")
                    logger.info(f"  Training:   loss={train_metrics['loss']:.4f}, accuracy={train_metrics['accuracy']:.4f}")
                    logger.info(f"  Validation: loss={val_loss:.4f}, accuracy={val_acc:.4f}")
                    logger.info(f"  Time: {epoch_time:.1f}s, No improvement: {self.epochs_without_improvement}/{self.early_stopping_patience}")
                    if improved:
                        logger.info(f"  New best model saved! (val_acc: {val_acc:.4f})")
                    
                    # Check for early stopping
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                    
                    # Log to wandb
                    self._log_to_wandb(epoch, train_metrics, val_metrics, epoch_time)
                    
                    # Check if RL intervention is needed
                    epochs_since_intervention = epoch + 1 - self.last_intervention_epoch
                    if epochs_since_intervention >= self.intervention_frequency:
                        if self._check_for_intervention():
                            self.last_intervention_epoch = epoch + 1
                        else:
                            logger.info(f"  No intervention needed at epoch {epoch+1}")
                else:
                    # Log progress without validation metrics - better formatted
                    logger.info(f"\n=== Epoch {epoch+1}/{self.max_epochs} ===")
                    logger.info(f"  Training: loss={train_metrics['loss']:.4f}, accuracy={train_metrics['accuracy']:.4f}")
                    logger.info(f"  Time: {epoch_time:.1f}s")
                    
                    # Log to wandb (training only)
                    self._log_to_wandb(epoch, train_metrics, None, epoch_time)
                
                # Save checkpoint if needed
                if (epoch + 1) % self.checkpoint_frequency == 0:
                    self._save_checkpoint()
                
                # Save history after each epoch
                self._save_history()
            
            # Final checkpoint and cleanup
            self._save_checkpoint()
            self._save_history()
            self._plot_training_history()
            
            total_time = time.time() - self.start_time
            logger.info(f"\n=== Training completed ===")
            logger.info(f"  Total time: {total_time:.1f}s")
            logger.info(f"  Best validation accuracy: {self.best_val_acc:.4f}")
            logger.info(f"  Total epochs: {self.current_epoch + 1}")
            logger.info(f"  RL interventions: {len(self.history['rl_interventions'])}")
            
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
            
            # Final save of RL brain at the end of training
            brain_final_path = os.path.join(
                self.log_dir, 
                'rl_brains', 
                f'brain_final.zip'
            )
            os.makedirs(os.path.dirname(brain_final_path), exist_ok=True)
            self.rl_optimizer.save_brain(brain_final_path)
            logger.info(f"Final RL brain saved to {brain_final_path}")
            
            return self.history
            
        except Exception as e:
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
        Determine if RL intervention is needed based on performance trends.
        
        Returns:
            bool: True if intervention is performed, False otherwise
        """
        # Only consider intervention after minimum number of epochs
        if (self.current_epoch + 1) < self.min_epochs_before_intervention:
            logger.info(f"Not intervening: Only at epoch {self.current_epoch+1}, need {self.min_epochs_before_intervention}")
            return False
            
        # Update the RL agent's CNN epoch awareness
        self.rl_optimizer.update_cnn_epoch(self.current_epoch)
        
        # Calculate performance trends using rolling windows
        trend_data = calculate_performance_trends({
            'history': self.history,
            'metric_window_size': self.metric_window_size,
            'improvement_threshold': self.improvement_threshold,
            'loss_stagnation_threshold': self.config_training.get('loss_stagnation_threshold', 0.003)
        })
        
        # Log trend data
        is_stagnating = trend_data['is_stagnating'] # TODO: it is never used to decide anything
        improvement_rate = trend_data['improvement_rate']
        logger.info(f"Performance trends: stagnating={is_stagnating}, improvement_rate={improvement_rate:.5f}")
        
        # Prepare observation for RL agent
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
        logger.info(f"RL agent intervening at epoch {self.current_epoch+1}")
        
        # Increment intervention counter
        self.intervention_count += 1
        
        # Store pre-intervention metrics for calculating reward later
        val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
        val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else float('inf')
        
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
        major_intervention = self._is_major_intervention(new_hyperparams) # TODO: check if this is correct
        
        if major_intervention:
            logger.info("Major intervention detected, restarting training from scratch")
            self._restart_training(new_hyperparams)
        else:
            logger.info("Minor intervention detected, continuing training")
            self._apply_hyperparams(new_hyperparams)
        
        # Record the intervention in history
        intervention = {
            'epoch': self.current_epoch + 1,
            'hyperparameters': new_hyperparams.copy(),
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
                logger.warning(f"Failed to log RL intervention to Weights & Biases: {str(e)}")
        
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
        
        # Track all major changes
        major_changes = []
        
        # Check each major hyperparameter for changes
        if new_hyperparams.get('fc_layers') != current_hyperparams.get('fc_layers'):
            major_changes.append(f"FC layer configuration changed from {current_hyperparams.get('fc_layers')} to {new_hyperparams.get('fc_layers')}")
            
        if new_hyperparams.get('optimizer_type') != current_hyperparams.get('optimizer_type'):
            major_changes.append(f"Optimizer type changed from {current_hyperparams.get('optimizer_type')} to {new_hyperparams.get('optimizer_type')}")
        
        # Log all detected changes
        if major_changes:
            for change in major_changes:
                logger.info(f"Major intervention: {change}")
            return True
        else:
            # No major changes detected
            logger.info("Minor intervention: Only learning rates, dropout, or weight decay changed")
            return False

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
            
            # Log detailed experience information
            logger.debug(f"Stored SARST experience with reward: {reward:.4f}, done: {experience['done']}")
        else:
            # Fallback to just storing the reward if we don't have complete data
            logger.warning("Incomplete intervention data, storing reward only")
            self.collected_experiences.append(reward)
        
        # Log the reward and episode status
        if len(self.collected_experiences) == 1:  # First experience in a new episode
            logger.info(f"Starting new RL episode ({len(self.collected_episodes) + 1})")
        
        logger.info(f"RL reward for {'intervention' if intervened else 'non-intervention'}: "
                   f"{reward:.4f} (acc change: {acc_improvement:.4f}, loss change: {loss_improvement:.4f})")
        
        # Check if we've completed an episode (5 interventions)
        if len(self.collected_experiences) >= 5:
            logger.info(f"Episode {len(self.collected_episodes) + 1} completed with {len(self.collected_experiences)} interventions")
            
            # Store the completed episode
            self.collected_episodes.append(self.collected_experiences)
            
            # Train the RL agent with collected experiences
            training_success = self.rl_optimizer.learn_from_episode(self.collected_experiences)
            
            # Clear the collected experiences after training
            self.collected_experiences = []
            
            if training_success:
                logger.info(f"RL agent trained successfully with episode data")
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
            
            logger.info(f"Checkpoint saved at epoch {self.current_epoch+1}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

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
            
            logger.info(f"Best model saved at epoch {self.current_epoch+1} with val_acc={self.best_val_acc:.4f}")
        except Exception as e:
            logger.error(f"Error saving best model: {str(e)}")

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
            
            logger.info(f"Resumed training from epoch {self.current_epoch}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def _plot_training_history(self) -> None:
        """Plot training history metrics."""
        try:
            if len(self.history['epoch']) < 2:
                return  # Not enough data to plot
                
            plt.figure(figsize=(15, 10))
            
            # Plot loss
            plt.subplot(2, 2, 1)
            plt.plot(self.history['epoch'], self.history['train_loss'], label='Train')
            plt.plot(self.history['epoch'], self.history['val_loss'], label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            # Plot accuracy
            plt.subplot(2, 2, 2)
            plt.plot(self.history['epoch'], self.history['train_acc'], label='Train')
            plt.plot(self.history['epoch'], self.history['val_acc'], label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            
            # Plot time per epoch
            plt.subplot(2, 2, 3)
            plt.plot(self.history['epoch'], self.history['time_per_epoch'])
            plt.xlabel('Epoch')
            plt.ylabel('Time (s)')
            plt.title('Time per Epoch')
            
            # Mark interventions on the accuracy plot
            if self.history['rl_interventions']:
                plt.subplot(2, 2, 2)
                intervention_epochs = [interv['epoch'] for interv in self.history['rl_interventions']]
                intervention_accs = []
                for ep in intervention_epochs:
                    # Find the closest match in history
                    if ep in self.history['epoch']:
                        idx = self.history['epoch'].index(ep)
                        intervention_accs.append(self.history['val_acc'][idx])
                    else:
                        # Find nearest epoch
                        nearest_idx = min(range(len(self.history['epoch'])), 
                                         key=lambda i: abs(self.history['epoch'][i] - ep))
                        intervention_accs.append(self.history['val_acc'][nearest_idx])
                        
                plt.scatter(intervention_epochs, intervention_accs, color='red', s=100, marker='x',
                           label='RL Interventions')
                plt.legend()
            
            # Plot hyperparameter changes if we have multiple sets
            if len(self.history['hyperparameters']) > 1:
                plt.subplot(2, 2, 4)
                
                # Only plot numeric hyperparameters
                numeric_params = ['learning_rate', 'weight_decay', 'dropout_rate']
                for param in numeric_params:
                    if all(param in hp for hp in self.history['hyperparameters']):
                        values = [hp[param] for hp in self.history['hyperparameters']]
                        
                        # For the x-axis, use the epochs when changes happened
                        x_values = [1]  # First epoch starts with initial hyperparams
                        if len(values) > 1:
                            x_values.extend([interv['epoch'] for interv in self.history['rl_interventions']])
                        
                        # Only plot if we have intervention data
                        if len(x_values) == len(values):
                            plt.plot(x_values, values, 'o-', label=param)
                
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.title('Hyperparameter Changes')
                plt.xscale('log') if any('learning_rate' in hp for hp in self.history['hyperparameters']) else None
                plt.legend()
            
            plt.tight_layout()
            plot_path = os.path.join(self.log_dir, 'training_history.png')
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Training history plots saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")

    def _signal_handler(self, sig, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {sig}, initiating graceful shutdown")

    def _setup_signal_handling(self):
        """Setup signal handlers for graceful termination."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _cleanup(self):
        """Cleanup resources before exit."""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            logger.info(f"Training session lasted {total_time:.1f}s")
        
        # Final save of history
        self._save_history()
        
        # Finish wandb run if active
        if self.wandb_initialized:
            try:
                # Log final summary if we haven't already
                wandb.log({
                    "final_epoch": self.current_epoch + 1,
                    "best_val_acc": self.best_val_acc,
                    "completed": not self.training_interrupted,
                    "early_stopped": self.epochs_without_improvement >= self.early_stopping_patience
                }, step=self.global_wandb_step)
                
                # Finish the run properly
                wandb.finish()
                logger.info("Weights & Biases run finished")
            except Exception as e:
                logger.warning(f"Error finalizing wandb run: {str(e)}")