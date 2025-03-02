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
        self.config = config
        
        # Get training parameters
        self.max_epochs = config['training'].get('max_epochs', 100)
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 25)
        self.eval_frequency = config['training'].get('eval_frequency', 1)
        self.checkpoint_frequency = config['training'].get('checkpoint_frequency', 5)
        self.min_epochs_before_intervention = config['training'].get('min_epochs_before_intervention', 5)
        
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
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_interrupted = False
        self.intervention_history = []
        self.start_time = None
        
        # Initialize history file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_file = os.path.join(self.log_dir, f"training_history_{timestamp}.json")
        # Also save a copy to a fixed location for easy access
        self.fixed_history_file = os.path.join(self.log_dir, "training_history.json")
        
        # Setup signal handling for graceful termination
        self._setup_signal_handling()
        
        # Initialize wandb
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.wandb_initialized = True
        
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
            
        # Initialize wandb before training starts
        if self.use_wandb and not self.wandb_initialized:
            self._init_wandb()
        
        # Track pre-intervention metrics
        self.last_intervention_metrics = None
            
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
                
                # Evaluate if needed
                if (epoch + 1) % self.eval_frequency == 0:
                    val_metrics = self.cnn_trainer.evaluate()
                else:
                    val_metrics = {'loss': None, 'accuracy': None}
                
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
                    
                    # Log progress
                    status_msg = (
                        f"Epoch {epoch+1}/{self.max_epochs}: "
                        f"train_loss={train_metrics['loss']:.4f}, "
                        f"train_acc={train_metrics['accuracy']:.4f}, "
                        f"val_loss={val_loss:.4f}, "
                        f"val_acc={val_acc:.4f}, "
                        f"time={epoch_time:.1f}s"
                    )
                    if improved:
                        status_msg += " (improved)"
                    logger.info(status_msg)
                    
                    # Check for early stopping
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                    
                    # Log to wandb
                    self._log_to_wandb(epoch, train_metrics, val_metrics, epoch_time)
                    
                    # Store current metrics before potential intervention
                    pre_intervention_metrics = {
                        'val_acc': val_acc,
                        'val_loss': val_loss
                    }
                    
                    # Check if RL intervention is needed
                    if self._check_for_intervention():
                        # Store pre-intervention metrics for reward calculation
                        self.last_intervention_metrics = pre_intervention_metrics
                else:
                    # Log progress without validation metrics
                    logger.info(
                        f"Epoch {epoch+1}/{self.max_epochs}: "
                        f"train_loss={train_metrics['loss']:.4f}, "
                        f"train_acc={train_metrics['accuracy']:.4f}, "
                        f"time={epoch_time:.1f}s"
                    )
                    
                    # Log to wandb (training only)
                    self._log_to_wandb(epoch, train_metrics, None, epoch_time)
                
                # Save checkpoint if needed
                if (epoch + 1) % self.checkpoint_frequency == 0:
                    self._save_checkpoint()
                
                # Save history after each epoch
                self._save_history()
                
                # Check if training should be interrupted
                if self.training_interrupted:
                    logger.info("Training interrupted by user")
                    break
            
            # Final checkpoint and cleanup
            self._save_checkpoint()
            self._save_history()
            self._plot_training_history()
            
            total_time = time.time() - self.start_time
            logger.info(f"Training completed in {total_time:.1f}s")
            
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
            
            # Log to wandb
            wandb.log(metrics, step=epoch + 1)
            
        except Exception as e:
            logger.warning(f"Failed to log to Weights & Biases: {str(e)}")

    def _check_for_intervention(self) -> bool:
        """
        Let the RL agent decide if intervention is needed and apply hyperparameter changes.
        
        Returns:
            bool: True if intervention occurred, False otherwise
        """
        # Only consider intervention after minimum number of epochs to allow for initial training
        if self.current_epoch < self.min_epochs_before_intervention:
            return False
            
        # Prepare observation for RL agent
        observation = self._prepare_observation()
        
        # Let the RL agent decide if intervention is needed
        new_hyperparams = self.rl_optimizer.optimize_hyperparameters(observation)
        
        # If RL agent returns None or empty dict, no intervention needed
        if not new_hyperparams:
            # Log that we considered an intervention but the agent declined
            logger.info(f"RL agent decided not to intervene at epoch {self.current_epoch+1}")
            
            # Track this decision not to intervene for learning
            if self.wandb_initialized:
                try:
                    wandb.log({
                        "rl_considered_intervention": True,
                        "rl_decided_to_intervene": False,
                        "rl_intervention_epoch": self.current_epoch + 1
                    }, step=self.current_epoch + 1)
                except Exception as e:
                    logger.warning(f"Failed to log RL non-intervention to W&B: {str(e)}")
                    
            # Store pre-intervention metrics for potential rewards
            # (even for non-interventions, so agent can learn if this was a good choice)
            if len(self.history['val_acc']) > 0:
                self.last_intervention_metrics = {
                    'val_acc': self.history['val_acc'][-1],
                    'val_loss': self.history['val_loss'][-1],
                    'intervened': False
                }
                
            return False
            
        # Apply new hyperparameters
        logger.info(f"RL agent intervening at epoch {self.current_epoch+1}")
        logger.info(f"New hyperparameters: {new_hyperparams}")
        
        # Store pre-intervention metrics for calculating reward later
        val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
        val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else float('inf')
        
        # Store metrics for reward calculation
        self.last_intervention_metrics = {
            'val_acc': val_acc,
            'val_loss': val_loss,
            'intervened': True
        }
        
        # Update model with new hyperparameters
        self.cnn_trainer.model.update_hyperparams(new_hyperparams)
        
        # Update optimizer
        self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()
        
        # Record intervention in history
        intervention = {
            'epoch': self.current_epoch + 1,
            'hyperparameters': new_hyperparams.copy(),
            'val_acc_before': val_acc
        }
        self.history['rl_interventions'].append(intervention)
        
        # Update hyperparameters in history
        self.history['hyperparameters'].append(new_hyperparams.copy())
        
        # Log intervention to wandb
        if self.wandb_initialized:
            try:
                # Log the intervention event
                wandb.log({
                    "rl_considered_intervention": True,
                    "rl_decided_to_intervene": True,
                    "rl_intervention_epoch": self.current_epoch + 1,
                    "val_acc_before_intervention": val_acc
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

    def _provide_rl_reward_after_training(self, pre_val_acc, pre_val_loss, intervened=True):
        """
        Calculate reward for RL agent and make it learn based on the intervention outcome.
        
        Args:
            pre_val_acc: Validation accuracy before intervention
            pre_val_loss: Validation loss before intervention
            intervened: Whether an intervention was actually performed
        """
        # Skip if we don't have recent validation metrics
        if not self.history['val_acc'] or not self.history['val_loss']:
            return
            
        # Get current metrics
        current_val_acc = self.history['val_acc'][-1]
        current_val_loss = self.history['val_loss'][-1]
        
        if intervened:
            # Calculate reward for an actual intervention
            # Reward improvement in accuracy and reduction in loss
            acc_improvement = current_val_acc - pre_val_acc
            loss_improvement = pre_val_loss - current_val_loss
            
            # Combine improvements into a reward
            # Weight accuracy improvement higher
            reward = acc_improvement * 10.0 + loss_improvement * 5.0
            
            # Ensure reward is not negative (minimum 0.1 to prevent discouraging exploration)
            reward = max(0.1, reward)
            
            logger.info(f"RL reward for intervention: {reward:.4f} (acc: {current_val_acc:.4f} vs {pre_val_acc:.4f}, "
                        f"loss: {current_val_loss:.4f} vs {pre_val_loss:.4f})")
        else:
            # For non-interventions, compute what would have happened
            # A small change or improvement means not intervening was good
            # A large degradation means intervention might have been better
            acc_change = abs(current_val_acc - pre_val_acc)
            loss_change = abs(current_val_loss - pre_val_loss)
            
            # If metrics got significantly worse, this might indicate we should have intervened
            # If metrics stayed stable or improved, not intervening was a good choice
            if acc_change < 0.01 and loss_change < 0.05:
                # Stable or small improvement - good choice not to intervene
                reward = 0.5  # Moderate positive reward
            else:
                if current_val_acc < pre_val_acc - 0.02:
                    # Accuracy got significantly worse - should have intervened
                    reward = 0.05  # Small reward (nearly neutral)
                else:
                    # Changes were acceptable without intervention
                    reward = 0.2  # Small positive reward
            
            logger.info(f"RL reward for non-intervention: {reward:.4f} (acc change: {acc_change:.4f}, "
                        f"loss change: {loss_change:.4f})")
        
        # Create new observation after intervention/non-intervention
        new_observation = self._prepare_observation()
        
        # Make RL agent learn from this outcome
        self.rl_optimizer.learn_from_intervention(new_observation, reward, intervened=intervened)
        
        # Periodically save the RL brain (e.g., every 5 decisions)
        decisions_count = len(self.history['rl_interventions'])
        save_frequency = self.config.get('rl_brain_save_frequency', 5)
        
        if decisions_count > 0 and decisions_count % save_frequency == 0:
            brain_path = os.path.join(
                self.log_dir, 
                'rl_brains', 
                f'brain_after_{decisions_count}_decisions.zip'
            )
            os.makedirs(os.path.dirname(brain_path), exist_ok=True)
            self.rl_optimizer.save_brain(brain_path)
            logger.info(f"RL brain saved after {decisions_count} decisions")

    def _prepare_observation(self) -> np.ndarray:
        """
        Prepare observation for RL agent with enhanced information.
        
        Returns:
            np.ndarray: Observation vector
        """
        # Enhanced observation space with more metrics to help the agent decide when to intervene
        observation = np.zeros(18, dtype=np.float32)
        
        # Fill with relevant metrics from history
        if self.history['val_acc']:
            observation[0] = self.history['val_acc'][-1]  # Current val accuracy
            observation[1] = min(1.0, max(0.0, 1.0 - self.history['val_loss'][-1] / 10))  # Normalized val loss
            observation[2] = self.history['train_acc'][-1] if self.history['train_acc'] else 0.0  # Train accuracy
            observation[3] = min(1.0, max(0.0, 1.0 - self.history['train_loss'][-1] / 10))  # Normalized train loss
            
            # Add trend information - last 3 epochs
            if len(self.history['val_acc']) >= 3:
                # Accuracy trends (positive = improving, negative = degrading)
                acc_trend1 = self.history['val_acc'][-1] - self.history['val_acc'][-2]
                acc_trend2 = self.history['val_acc'][-2] - self.history['val_acc'][-3]
                observation[4] = acc_trend1  # Most recent change
                observation[5] = acc_trend2  # Previous change
                
                # Loss trends (positive = degrading, negative = improving)
                loss_trend1 = self.history['val_loss'][-1] - self.history['val_loss'][-2]
                loss_trend2 = self.history['val_loss'][-2] - self.history['val_loss'][-3]
                observation[6] = -loss_trend1  # Negative so positive = improving
                observation[7] = -loss_trend2  # Negative so positive = improving
        
        # Best performance so far
        observation[8] = self.best_val_acc  # Best validation accuracy
        observation[9] = min(1.0, max(0.0, 1.0 - self.best_val_loss / 10))  # Normalized best val loss
        
        # Current hyperparameters (normalized)
        current_hyperparams = self._get_current_hyperparams()
        
        # Learning rate (log scale normalization)
        lr = current_hyperparams.get('learning_rate', 0.001)
        observation[10] = (np.log10(lr) + 5) / 3  # Log scale from 1e-5 to 1e-2
            
        # Weight decay (log scale normalization)
        wd = current_hyperparams.get('weight_decay', 1e-4)
        observation[11] = (np.log10(wd) + 6) / 4  # Log scale from 1e-6 to 1e-2
            
        # Dropout rate (linear scale)
        observation[12] = current_hyperparams.get('dropout_rate', 0.5)
            
        # Optimizer type (one-hot like encoding)
        opt_type = current_hyperparams.get('optimizer_type', 'adam')
        if opt_type == 'adam':
            observation[13] = 0.33
        elif opt_type == 'sgd':
            observation[14] = 0.33
        else:  # adamw
            observation[15] = 0.33
        
        # Training progress
        observation[16] = self.current_epoch / self.max_epochs  # Progress through training
        
        # Epochs without improvement (normalized)
        observation[17] = self.epochs_without_improvement / self.early_stopping_patience
        
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
        
        # Check if there was a recent intervention (within the last epoch)
        if (hasattr(self, 'last_intervention_metrics') and 
            self.last_intervention_metrics is not None):
            # Calculate reward and provide feedback to RL agent
            self._provide_rl_reward_after_training(
                self.last_intervention_metrics['val_acc'],
                self.last_intervention_metrics['val_loss'],
                intervened=self.last_intervention_metrics.get('intervened', True)
            )
            # Clear the last intervention metrics after providing reward
            self.last_intervention_metrics = None

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

    def _signal_handler(self, sig, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {sig}, initiating graceful shutdown")
        self.training_interrupted = True

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
                })
                
                # Mark run as crashed if training was interrupted
                if self.training_interrupted:
                    wandb.mark_preempting()
                
                # Finish the run properly
                wandb.finish()
                logger.info("Weights & Biases run finished")
            except Exception as e:
                logger.warning(f"Error finalizing wandb run: {str(e)}")