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
from collections import deque

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
        self.config = config['training']
        
        # Get training parameters
        self.max_epochs = self.config.get('max_epochs', 100)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.eval_frequency = self.config.get('eval_frequency', 1)
        self.checkpoint_frequency = self.config.get('checkpoint_frequency', 5)
        
        # Add save_freq attribute that matches checkpoint_frequency
        self.save_freq = self.config.get('save_freq', self.checkpoint_frequency)
        
        # Setup logging
        self.log_dir = config['logging'].get('log_dir', 'logs')
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
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_interrupted = False
        self.intervention_history = []
        self.start_time = None
        
        # Initialize history file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_file = os.path.join(self.log_dir, f"training_history_{timestamp}.json")
        
        # Initialize wandb
        self.use_wandb = self.config.get('use_wandb', True)
        self.wandb_initialized = False

        # Extract RL intervention settings
        self.min_epochs_before_intervention = self.config.get("min_epochs_before_intervention", 5)
        self.intervention_frequency = self.config.get("intervention_frequency", 3)
        self.stagnation_threshold = self.config.get("stagnation_threshold", 0.005)
        

    def train(self, epochs=None, steps=None):
        """
        Run the training loop with RL-based hyperparameter optimization.
        
        Args:
            epochs (int, optional): Number of epochs to train for. If None, uses the
                                   value from config or defaults to class attribute.
            steps (int, optional): Number of steps per epoch. If None, uses all data.
                                  
        Returns:
            dict: Training history metrics
        """
        logger.info("Starting training with RL-based hyperparameter optimization.")
        self.training_start_time = time.time()
        
        # Set max_epochs from passed parameter if provided
        if epochs is not None:
            self.max_epochs = epochs
            logger.info(f"Using provided epochs value: {self.max_epochs}")
        
        # Main training loop
        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            logger.info(f"Epoch {epoch}/{self.max_epochs}")
            train_metrics = self._train_epoch()
            
            # Evaluate on validation set
            val_metrics = self.cnn_trainer.evaluate()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update history with this epoch's metrics
            self._update_history(epoch, train_metrics, val_metrics, epoch_time)
            
            # Convert metrics to float before logging
            try:
                train_loss = float(train_metrics['loss'])
                train_acc = float(train_metrics['accuracy'])
                val_loss = float(val_metrics['loss'])
                val_acc = float(val_metrics['accuracy'])

                # Log to console
                logger.info(f"Epoch {epoch}/{self.max_epochs} - "
                        f"train_loss: {train_loss:.4f}, "
                        f"train_acc: {train_acc:.4f}, "
                        f"val_loss: {val_loss:.4f}, "
                        f"val_acc: {val_acc:.4f}, "
                        f"time: {epoch_time:.2f}s")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting metrics to float: {e}")
                train_loss = train_metrics['loss']
                train_acc = train_metrics['accuracy']
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']

                # Log to console
                logger.info(f"Epoch {epoch}/{self.max_epochs} - "
                        f"train_loss: {train_loss}, "
                        f"train_acc: {train_acc}, "
                        f"val_loss: {val_loss}, "
                        f"val_acc: {val_acc}, "
                        f"time: {epoch_time}s")
            
            # Log to wandb
            if self.use_wandb:
                self._log_to_wandb(epoch, train_metrics, val_metrics, epoch_time)
            
            # Track best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.epochs_without_improvement = 0
                self._save_best_model()
                self.best_val_loss = val_loss
            else:
                self.epochs_without_improvement += 1
            
            # Check for early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs without improvement")
                break
            
            # Check for RL-based hyperparameter intervention
            if self._should_intervene():
                self._check_for_intervention()
            
            # Save checkpoint if needed
            if epoch % self.checkpoint_frequency == 0:
                self._save_checkpoint(epoch)
            
            # Save history after each epoch
            self._save_history()
            
            # Check for interruption
            if self.training_interrupted:
                logger.info("Training interrupted by user")
                break
        
        # Save final model
        self._save_checkpoint(epoch, is_final=True)
        
        # Log training summary
        self._log_training_summary()

        # Plot training history
        self._plot_training_history()
        
        return self.history

    def _log_training_summary(self):
        """Log a summary of the training process."""
        logger.info("Training Summary:")
        logger.info(f"Total epochs: {self.current_epoch}")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Total RL interventions: {len(self.history['rl_interventions'])}")
        logger.info(f"Training interrupted: {self.training_interrupted}")
        logger.info(f"Early stopping triggered: {self.epochs_without_improvement >= self.early_stopping_patience}")

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
                "training/best_val_accuracy": self.best_val_accuracy,
            })
            
            # Log to wandb
            wandb.log(metrics, step=epoch + 1)
            
        except Exception as e:
            logger.warning(f"Failed to log to Weights & Biases: {str(e)}")

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for a single epoch.
        
        Returns:
            dict: Training metrics for this epoch
        """
        try:
            # Delegate to CNN trainer
            train_loss, train_acc = self.cnn_trainer.train_epoch().values()
            
            return {
                'loss': train_loss,
                'accuracy': train_acc
            }
        except Exception as e:
            logger.error(f"Error in epoch {self.current_epoch+1}: {str(e)}")
            # Save state before propagating error
            self._save_history()
            raise

    def _check_for_intervention(self) -> bool:
        """
        Check if RL agent should intervene and apply hyperparameter changes.
        
        Returns:
            bool: True if intervention occurred, False otherwise
        """
        # Only consider intervention after minimum number of epochs
        if self.current_epoch < self.min_epochs_before_intervention:
            return False
            
        # Get validation history for RL agent
        val_acc_history = self.history['val_acc']
        val_loss_history = self.history['val_loss']
        
        # Skip if we don't have enough history
        if len(val_acc_history) < self.min_epochs_before_intervention:
            return False
            
        # Prepare observation for RL agent
        observation = self._prepare_observation()
        
        # Ask RL agent if intervention is needed
        new_hyperparams = self.rl_optimizer.optimize_hyperparameters(
            observation, val_acc_history, val_loss_history
        )
        
        # If RL agent returns None or empty dict, no intervention needed
        if not new_hyperparams:
            return False
            
        # Apply new hyperparameters
        logger.info(f"RL agent intervening at epoch {self.current_epoch+1}")
        logger.info(f"New hyperparameters: {new_hyperparams}")
        
        # Update model with new hyperparameters
        self.cnn_trainer.model.update_hyperparams(new_hyperparams)
        
        # Update optimizer
        self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()
        
        # Record intervention in history
        intervention = {
            'epoch': self.current_epoch + 1,
            'hyperparameters': new_hyperparams.copy(),
            'val_acc_before': val_acc_history[-1]
        }
        self.history['rl_interventions'].append(intervention)
        
        # Update hyperparameters in history
        self.history['hyperparameters'].append(new_hyperparams.copy())
        
        # Log intervention to wandb
        if self.wandb_initialized:
            try:
                # Log the intervention event
                wandb.log({
                    "rl_intervention": True,
                    "rl_intervention_epoch": self.current_epoch + 1,
                    "val_acc_before_intervention": self.history['val_acc'][-1]
                }, step=self.current_epoch + 1)
                
                # Log the hyperparameter changes
                changes = {}
                for param_name, new_value in new_hyperparams.items():
                    # For complex structures like fc_layers, convert to string for logging
                    if not isinstance(new_value, (int, float, str, bool)):
                        new_value = str(new_value)
                    changes[f"rl_changes/{param_name}"] = new_value
                
                wandb.log(changes, step=self.current_epoch + 1)
                
            except Exception as e:
                logger.warning(f"Failed to log RL intervention to Weights & Biases: {str(e)}")
        
        return True

    def _prepare_observation(self) -> np.ndarray:
        """
        Prepare observation for RL agent.
        
        Returns:
            np.ndarray: Observation vector
        """
        # This should match the observation space defined in the RL environment
        observation = np.zeros(15, dtype=np.float32)
        
        # Fill with relevant metrics from history
        if self.history['val_acc']:
            observation[0] = self.history['val_acc'][-1]  # Current val accuracy
            observation[1] = min(1.0, max(0.0, 1.0 - self.history['val_loss'][-1] / 10))  # Normalized val loss
            observation[2] = self.history['train_acc'][-1] if self.history['train_acc'] else 0.0  # Train accuracy
            observation[3] = min(1.0, max(0.0, 1.0 - self.history['train_loss'][-1] / 10))  # Normalized train loss
            
        # Best performance so far
        observation[4] = self.best_val_accuracy  # Best validation accuracy
        observation[5] = min(1.0, max(0.0, 1.0 - self.best_val_loss / 10))  # Normalized best val loss
        
        # Current hyperparameters (normalized)
        current_hyperparams = self._get_current_hyperparams()
        
        # Learning rate (log scale normalization)
        lr = current_hyperparams.get('learning_rate', 0.001)
        observation[6] = (np.log10(lr) + 5) / 3  # Log scale from 1e-5 to 1e-2
            
        # Weight decay (log scale normalization)
        wd = current_hyperparams.get('weight_decay', 1e-4)
        observation[7] = (np.log10(wd) + 6) / 4  # Log scale from 1e-6 to 1e-2
            
        # Dropout rate (linear scale)
        observation[8] = current_hyperparams.get('dropout_rate', 0.5)
            
        # Optimizer type (one-hot like encoding)
        opt_type = current_hyperparams.get('optimizer_type', 'adam')
        if opt_type == 'adam':
            observation[9] = 0.33
        elif opt_type == 'sgd':
            observation[10] = 0.33
        else:  # adamw
            observation[11] = 0.33
        
        # Training progress
        observation[12] = self.current_epoch / self.max_epochs  # Progress through training
        
        # Epochs without improvement (normalized)
        observation[13] = self.epochs_without_improvement / self.early_stopping_patience
        
        # Relative improvement from last epoch
        if len(self.history['val_acc']) > 1:
            last_acc = self.history['val_acc'][-2]
            current_acc = self.history['val_acc'][-1]
            rel_improvement = (current_acc - last_acc) / max(0.01, last_acc)
            observation[14] = min(1.0, max(0.0, rel_improvement + 0.5))  # Scale to [0,1]
            
        return observation

    def _get_current_hyperparams(self) -> Dict[str, Any]:
        """
        Get current hyperparameters from model.
        
        Returns:
            dict: Current hyperparameters
        """
        if hasattr(self.cnn_trainer.model, 'hyperparams'):
            return self.cnn_trainer.model.hyperparams
        else:
            # Fallback to extracting params manually
            return {
                'learning_rate': self.cnn_trainer.optimizer.param_groups[0]['lr'],
                'weight_decay': self.cnn_trainer.optimizer.param_groups[0]['weight_decay'],
                # Default values for other params
                'dropout_rate': 0.5,
                'optimizer_type': type(self.cnn_trainer.optimizer).__name__.lower()
            }

    def _update_history(self, epoch: int, train_metrics: Dict[str, float], 
                        val_metrics: Union[Dict[str, float], Tuple[float, float]], epoch_time: float) -> None:
        """
        Update training history with latest metrics.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict or tuple
            epoch_time: Time taken for this epoch
        """
        # Handle train metrics (which could be tuple or dict)
        if isinstance(train_metrics, tuple):
            train_loss, train_acc = train_metrics
        elif isinstance(train_metrics, dict):
            train_loss = train_metrics.get('loss', 0)
            train_acc = train_metrics.get('accuracy', 0)
        else:
            train_loss, train_acc = 0, 0
            
        # Handle validation metrics (which could be tuple or dict)
        if isinstance(val_metrics, tuple):
            val_loss, val_acc = val_metrics
        elif isinstance(val_metrics, dict):
            val_loss = val_metrics.get('loss', 0)
            val_acc = val_metrics.get('accuracy', 0)
        else:
            val_loss, val_acc = 0, 0
            
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['time_per_epoch'].append(epoch_time)
        self.history['timestamp'].append(datetime.now().isoformat())

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
            logger.debug(f"Training history saved to {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")

    def _save_checkpoint(self, epoch, is_final=False) -> None:
        """Save training checkpoint."""
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.cnn_trainer.model.state_dict(),
                'optimizer_state_dict': self.cnn_trainer.optimizer.state_dict(),
                'best_val_accuracy': self.best_val_accuracy,
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
                'val_acc': self.best_val_accuracy,
                'val_loss': self.best_val_loss,
                'hyperparams': self._get_current_hyperparams()
            }, best_model_path)
            
            logger.info(f"Best model saved at epoch {self.current_epoch+1} with val_acc={self.best_val_accuracy:.4f}")
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
            self.best_val_accuracy = checkpoint['best_val_accuracy']
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
                intervention_accs = [self.history['val_acc'][self.history['epoch'].index(ep)] 
                                    for ep in intervention_epochs]
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
                    "best_val_accuracy": self.best_val_accuracy,
                    "completed": not self.training_interrupted,
                    "early_stopped": self.epochs_without_improvement >= self.early_stopping_patience
                })
                
                # Mark run as crashed if training was interrupted
                if self.training_interrupted:
                    wandb.mark_preempting()
                
                # Finish the run properly
                wandb.finish()
                logger.info("Weights & Biases run finished")
            except Exception as e:
                logger.warning(f"Error finalizing wandb run: {str(e)}")

    def _is_training_stagnated(self):
        """Check if training progress has stagnated based on validation accuracy"""
        val_acc_history = self.history['val_acc']
        if len(val_acc_history) < 4:
            return False
            
        # Calculate improvement over last 3 validation accuracies
        improvements = [val_acc_history[-i] - val_acc_history[-i-1] 
                for i in range(1, 4)]
        avg_improvement = sum(improvements) / len(improvements)
        
        # Training is stagnated if average improvement is below threshold
        logger.info(f"Average improvement over last 3 epochs: {avg_improvement:.4f} and threshold: {self.stagnation_threshold}")
        return avg_improvement < self.stagnation_threshold
    
    def _should_intervene(self):
        """Determine if the RL agent should intervene in the training process"""
        # No intervention before minimum number of epochs
        if self.current_epoch < self.min_epochs_before_intervention:
            logger.info(f"Skipping intervention before epoch {self.min_epochs_before_intervention}")
            return False
            
        # Only check on epochs that are multiples of intervention_frequency after min_epochs
        if (self.current_epoch - self.min_epochs_before_intervention) % self.intervention_frequency != 0:
            return False
            
        # Check if training is stagnated
        return self._is_training_stagnated()
