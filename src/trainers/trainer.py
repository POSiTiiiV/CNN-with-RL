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
        
        # Parameters for improved RL intervention strategy
        self.val_loss_window = self.config_training.get('val_loss_window', 5)
        self.loss_stagnation_threshold = self.config_training.get('loss_stagnation_threshold', 0.003)
        
        # NEW: Performance trend monitoring
        self.metric_window_size = self.config_training.get('metric_window_size', 5)  # Rolling window size
        self.improvement_threshold = self.config_training.get('improvement_threshold', 0.002)  # Minimum expected improvement
        
        # NEW: Gradual autonomy parameters
        self.initial_autonomy = self.config_rl.get('initial_autonomy', 0.2)  # Initially follow rules more
        self.max_autonomy = self.config_rl.get('max_autonomy', 0.9)  # Eventually, trust RL more
        self.current_autonomy = self.initial_autonomy  
        self.autonomy_increase_rate = self.config_rl.get('autonomy_increase_rate', 0.05)  # Rate of increase after successful interventions
        self.successful_interventions = 0
        
        # Intervention control and episode collection
        self.intervention_count = 0
        self.intervention_frequency = self.config_training.get('intervention_frequency', 5)
        self.last_intervention_epoch = 0
        self.epochs_without_improvement_threshold = self.config_training.get('epochs_without_improvement_threshold', 3)
        self.collected_intervention_results = []
        self.min_interventions_for_learning = self.config_rl.get('min_interventions_for_learning', 5)  # Collect at least 5 interventions
        
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
        self.training_interrupted = False
        self.start_time = None
        self.last_intervention_metrics = None
        
        # Initialize history file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_file = os.path.join(self.log_dir, f"training_history_{timestamp}.json")
        # Fixed location for training history for easy access
        training_history_dir = 'training_history'
        os.makedirs(training_history_dir, exist_ok=True)
        self.fixed_history_file = os.path.join(training_history_dir, f"training_history_{timestamp}.json")
        
        # Setup signal handling for graceful termination
        self._setup_signal_handling()
        
        # Initialize wandb
        self.use_wandb = self.config_logging.get('use_wandb', True)
        self.wandb_initialized = True
        
        # NEW: Track the global wandb step to prevent monotonicity issues
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
                    
                    # Check if RL intervention is needed
                    if (epoch + 1) % self.intervention_frequency == 0:  # TODO: instead of checking for intervention after every 'intervention_frequency' epochs(i,e., (current_epoch+1) % intervention_frequency == 0), we should check after 'intervention_frequency' number of epochs have passed since last the intervention
                        if self._check_for_intervention():
                            self.last_intervention_epoch = epoch + 1 # TODO: add a else statement telling the user how long till intervention
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
        Determine if intervention is needed based on performance trends.
        Uses a mix of rule-based and RL-based decisions with increasing autonomy.
        """
        # Only consider intervention after minimum number of epochs
        if (self.current_epoch+1) < self.min_epochs_before_intervention:
            logger.info(f"Not intervening: Only at epoch {self.current_epoch+1}, need {self.min_epochs_before_intervention}")
            return False
            
        # Update the RL agent's CNN epoch awareness
        self.rl_optimizer.update_cnn_epoch(self.current_epoch)
        
        # Calculate performance trends using rolling windows
        trend_data = calculate_performance_trends({
            'history': self.history,
            'metric_window_size': self.metric_window_size,
            'improvement_threshold': self.improvement_threshold,
            'loss_stagnation_threshold': self.loss_stagnation_threshold
        })
        is_stagnating = trend_data['is_stagnating']
        improvement_rate = trend_data['improvement_rate']
        
        # Only consider intervention if the model is stagnating or not improving fast enough
        if not is_stagnating and improvement_rate > self.improvement_threshold:
            logger.info(f"Training still progressing well (improvement rate: {improvement_rate:.5f})")
            return False
            
        # Log trend data
        logger.info(f"Performance trends: stagnating={is_stagnating}, " +
                    f"improvement_rate={improvement_rate:.5f}, " +
                    f"autonomy_level={self.current_autonomy:.2f}")
        
        # Mixed decision strategy based on current autonomy level
        # Lower autonomy = more rule-based decisions
        # Higher autonomy = more RL-based decisions
        rule_based_decision = is_stagnating or improvement_rate < self.improvement_threshold
        
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
        rl_decided_to_intervene = new_hyperparams is not None
        
        # Make final decision based on weighted combination of rule and RL
        random_factor = np.random.random()  # TODO: it looks like its supposed to be explore/exploit but i dont think the implementation is correct/complete
        should_intervene = (
            (random_factor > self.current_autonomy and rule_based_decision) or  # Rule-based with (1-autonomy) probability
            (random_factor <= self.current_autonomy and rl_decided_to_intervene)  # RL-based with autonomy probability
        )
        
        # Log the decision factors
        logger.info(f"Intervention decision: rule={rule_based_decision}, RL={rl_decided_to_intervene}, " +
                    f"final={should_intervene}, autonomy={self.current_autonomy:.2f}")
        
        if not should_intervene or not new_hyperparams:
            # Store metrics for non-intervention case
            if len(self.history['val_acc']) > 0:
                self.last_intervention_metrics = {
                    'val_acc': self.history['val_acc'][-1],
                    'val_loss': self.history['val_loss'][-1],
                    'intervened': False,
                    'epoch': self.current_epoch,
                    'trend_data': trend_data  # Store trend data for later reward calculation
                }
            return False
            
        # Apply new hyperparameters
        logger.info(f"RL agent intervening at epoch {self.current_epoch+1}")
        logger.info(f"New hyperparameters: {new_hyperparams}")
        
        # Increment intervention counter
        self.intervention_count += 1
        self.last_intervention_epoch = self.current_epoch+1
        
        # Store pre-intervention metrics for calculating reward later
        val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
        val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else float('inf')
        
        # Store metrics for reward calculation
        self.last_intervention_metrics = {
            'val_acc': val_acc,
            'val_loss': val_loss,
            'intervened': True,
            'epoch': self.current_epoch,
            'hyperparams': new_hyperparams.copy(),
            'trend_data': trend_data  # Store trend data for later reward calculation
        }
        
        # Determine if the intervention is major or minor
        major_intervention = self._is_major_intervention(new_hyperparams)
        
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
        
        if self.wandb_initialized:
            try:
                # Use global step counter instead of epoch
                self.global_wandb_step += 1
                current_step = self.global_wandb_step
                
                wandb.log({
                    "rl_considered_intervention": True,
                    "rl_decided_to_intervene": True,
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
        major_keys = ['fc_config', 'optimizer_type']
        for key in major_keys:
            if key in new_hyperparams and new_hyperparams[key] != self._get_current_hyperparams().get(key):
                return True
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
        
        # Don't reset global_wandb_step when restarting training
        # This ensures wandb logging remains monotonic
        
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

    def _apply_hyperparams(self, new_hyperparams: Dict[str, Any]) -> None:
        """
        Apply new hyperparameters without restarting training.
        
        Args:
            new_hyperparams: New hyperparameters proposed by the RL agent
        """
        self.cnn_trainer.model.update_hyperparams(new_hyperparams)
        self.cnn_trainer.optimizer = self.cnn_trainer.model.get_optimizer()

    def _check_loss_stability(self) -> bool:
        """
        Check if validation loss has stabilized (not improving significantly).
        
        Returns:
            bool: True if loss is stable, False otherwise
        """
        # Make sure we have enough history
        if len(self.history['val_loss']) < self.val_loss_window:
            return False
            
        # Get the recent validation loss values
        recent_losses = self.history['val_loss'][-self.val_loss_window:]
        
        # Calculate average change in loss
        changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
        avg_change = sum(changes) / len(changes)
        
        # If average change is below threshold, consider loss stable
        is_stable = avg_change < self.loss_stagnation_threshold
        
        if is_stable:
            logger.info(f"Validation loss stabilized with avg change of {avg_change:.5f}")
            
        return is_stable

    def _provide_rl_reward_after_training(self, pre_val_acc, pre_val_loss, intervened=True):
        """
        Calculate reward for RL agent based on long-term intervention impact.
        Considers performance trends before and after intervention.
        """
        # Skip if we don't have recent validation metrics
        if not self.history['val_acc'] or not self.history['val_loss']:
            return
            
        # Get current metrics
        current_val_acc = self.history['val_acc'][-1]
        current_val_loss = self.history['val_loss'][-1]
        
        # Calculate current trends to compare with pre-intervention trends
        current_trends = calculate_performance_trends({
            'history': self.history,
            'metric_window_size': self.metric_window_size,
            'improvement_threshold': self.improvement_threshold,
            'loss_stagnation_threshold': self.loss_stagnation_threshold
        })
        
        # Extract pre-intervention trends if available
        pre_trends = self.last_intervention_metrics.get('trend_data', None)
        
        if intervened:
            # Calculate immediate impact
            acc_improvement = current_val_acc - pre_val_acc
            loss_improvement = pre_val_loss - current_val_loss

            # Normalize improvements to prevent excessive scaling
            acc_improvement_norm = acc_improvement / (pre_val_acc + 1e-6)  # Avoid division by zero
            loss_improvement_norm = loss_improvement / (pre_val_loss + 1e-6)

            # Scale accuracy and loss contribution equally
            immediate_reward = (acc_improvement_norm * 7.5 + loss_improvement_norm * 7.5) * 0.6

            # Calculate trend-based improvement (scale dynamically)
            trend_improvement = 0
            if pre_trends:
                trend_diff = current_trends['improvement_rate'] - pre_trends.get('improvement_rate', 0)
                trend_improvement = np.clip(trend_diff * 10.0, -5.0, 5.0)  # Scale dynamically with limits

            # Weight trends more in long-term training
            reward = immediate_reward + (trend_improvement * 0.4)

            # Allow negative rewards for worsening trends
            if acc_improvement <= 0 and loss_improvement <= 0:
                reward -= 0.2  # Small penalty if both worsen

            # Prevent masking bad decisions with a minimum reward
            if reward < 0:
                reward = 0.0

            # Update autonomy level based on success
            self._update_autonomy(reward)

            
            logger.info(f"RL reward for intervention: {reward:.4f} (acc: {acc_improvement:.4f}, " +
                        f"loss: {loss_improvement:.4f}, trend: {trend_improvement:.4f})")
        else:
            # For non-interventions, assess if the decision was good
            # Compare current performance with expected performance had we intervened
            # FIX: Assign the return value to the reward variable
            reward = self._calculate_non_intervention_reward(pre_val_acc, pre_val_loss, current_val_acc, current_val_loss)
        
        # Store intervention result with trends
        intervention_result = {
            'pre_val_acc': pre_val_acc,
            'pre_val_loss': pre_val_loss,
            'post_val_acc': current_val_acc,
            'post_val_loss': current_val_loss,
            'reward': reward,  # Now reward is defined in both branches
            'intervened': intervened,
            'epoch': self.current_epoch,
            'pre_trends': pre_trends,
            'post_trends': current_trends
        }
        self.collected_intervention_results.append(intervention_result)
        
        # Train the RL agent if we have enough data
        if len(self.collected_intervention_results) >= self.min_interventions_for_learning:
            self._train_rl_from_collected_results()
    
    def _update_autonomy(self, reward: float):
        """
        Update the autonomy level based on intervention success.
        Successful interventions increase RL agent's autonomy.
        
        Args:
            reward: The reward received for the intervention
        """
        # Consider intervention successful if reward exceeds threshold
        if reward > 0.5:  # Threshold for good intervention
            self.successful_interventions += 1
            # Increase autonomy gradually
            autonomy_increase = self.autonomy_increase_rate * (reward / 2.0)  # Scale by reward
            self.current_autonomy = min(self.max_autonomy, 
                                       self.current_autonomy + autonomy_increase)
            logger.info(f"Intervention successful, increased autonomy to {self.current_autonomy:.3f}")
        else:
            # Decrease autonomy slightly for unsuccessful interventions
            self.current_autonomy = max(self.initial_autonomy, 
                                      self.current_autonomy - 0.01)
            logger.info(f"Intervention less successful, reduced autonomy to {self.current_autonomy:.3f}")
            
    def _train_rl_from_collected_results(self):
        """Train the RL agent from collected intervention results."""
        logger.info(f"Training RL agent with {len(self.collected_intervention_results)} intervention results")
        
        # Extract rewards from collected interventions
        rewards = [result['reward'] for result in self.collected_intervention_results]
        
        # Train the RL agent with all collected rewards
        training_successful = self.rl_optimizer.learn_from_episode(rewards)
        
        if training_successful:
            logger.info("RL agent training successful")
            
            # Clear collected interventions
            self.collected_intervention_results = []
            
            # Save the RL brain periodically
            decisions_count = len(self.history['rl_interventions'])
            brain_path = os.path.join(
                self.log_dir, 
                'rl_brains', 
                f'brain_after_{decisions_count}_decisions.zip'
            )
            os.makedirs(os.path.dirname(brain_path), exist_ok=True)
            self.rl_optimizer.save_brain(brain_path)
            logger.info(f"RL brain saved after {decisions_count} decisions")
        else:
            logger.warning("RL agent training failed")

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
        train_loss = train_metrics.get('loss', 0)
        train_acc = train_metrics.get('accuracy', 0)
        val_loss = val_metrics.get('loss', 0)
        val_acc = val_metrics.get('accuracy', 0)
            
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['time_per_epoch'].append(epoch_time)
        self.history['timestamp'].append(datetime.now().isoformat())
        
        # Check if there was a recent intervention
        if self.last_intervention_metrics is not None:
            # Add safety check in case epoch key is missing
            if 'epoch' in self.last_intervention_metrics:
                epochs_since_intervention = self.current_epoch - self.last_intervention_metrics['epoch']
            else:
                # If epoch key is missing, default to waiting 5 epochs
                self.last_intervention_metrics['epoch'] = self.current_epoch - 5
                epochs_since_intervention = 5
                logger.warning("Missing 'epoch' key in intervention metrics, defaulting to 5 epochs waiting period")
            
            # Wait at least 5 epochs to properly evaluate the effect of an intervention
            if epochs_since_intervention >= 5:
                # Calculate reward and provide feedback to RL agent
                self._provide_rl_reward_after_training(
                    self.last_intervention_metrics['val_acc'],
                    self.last_intervention_metrics['val_loss'],
                    intervened=self.last_intervention_metrics['intervened']
                )
                # Clear the intervention metrics
                self.last_intervention_metrics = None

    def _calculate_non_intervention_reward(self, pre_val_acc, pre_val_loss, current_val_acc, current_val_loss):
        """
        Calculate the reward for not intervening.
        
        Args:
            pre_val_acc: Validation accuracy before potential intervention
            pre_val_loss: Validation loss before potential intervention
            current_val_acc: Current validation accuracy
            current_val_loss: Current validation loss
            
        Returns:
            float: Reward value
        """
        # For non-interventions, compute what would have happened
        # A small change or improvement means not intervening was good
        # A large degradation means intervention might have been better
        acc_change = abs(current_val_acc - pre_val_acc)
        loss_change = abs(current_val_loss - pre_val_loss)
        
        # If metrics got significantly worse, this might indicate we should have intervened
        # If metrics stayed stable or improved, not intervening was a good choice
        if current_val_acc > pre_val_acc or current_val_loss < pre_val_loss:
            # Metrics improved without intervention - good decision
            reward = 0.5  # Moderate positive reward
        else:
            # Metrics worsened - maybe should have intervened
            if current_val_acc < pre_val_acc - 0.02:
                # Accuracy got significantly worse - should have intervened
                reward = 0.05  # Small reward (nearly neutral)
            else:
                # Changes were acceptable without intervention
                reward = 0.2  # Small positive reward
        
        logger.info(f"RL reward for non-intervention: {reward:.4f} (acc change: {acc_change:.4f}, "
                    f"loss change: {loss_change:.4f})")
        
        return reward
    
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
                # Use global step for final logging
                self.global_wandb_step += 1
                
                # Log final summary if we haven't already
                wandb.log({
                    "final_epoch": self.current_epoch + 1,
                    "best_val_acc": self.best_val_acc,
                    "completed": not self.training_interrupted,
                    "early_stopped": self.epochs_without_improvement >= self.early_stopping_patience
                }, step=self.global_wandb_step)
                
                # Mark run as crashed if training was interrupted
                if self.training_interrupted:
                    wandb.mark_preempting()
                
                # Finish the run properly
                wandb.finish()
                logger.info("Weights & Biases run finished")
            except Exception as e:
                logger.warning(f"Error finalizing wandb run: {str(e)}")