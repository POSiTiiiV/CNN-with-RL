import os
import time
import logging
import numpy as np
import torch
import json
from datetime import datetime
import wandb

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Alternative implementation of the ModelTrainer class with enhanced hyperparameter logging.
    """
    
    def __init__(self, cnn_trainer, rl_agent, config, output_dir=None, use_wandb=False):
        # ...initialization code would go here...
        self.cnn_trainer = cnn_trainer
        self.rl_agent = rl_agent
        self.config = config
        self.use_wandb = use_wandb
        self.output_dir = output_dir or "./output"
        
        # Create output directory for hyperparameter logs
        self.hp_log_dir = os.path.join(self.output_dir, "hyperparameter_logs")
        os.makedirs(self.hp_log_dir, exist_ok=True)
        
        # Hyperparameter log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hp_log_file = os.path.join(self.hp_log_dir, f"hyperparameter_changes_{timestamp}.json")
        
    def train(self, epochs=None, steps=None):
        """
        Training method with enhanced hyperparameter logging
        
        Args:
            epochs (int, optional): Number of epochs to train for. If None, uses the
                                   value from config or defaults to class attribute.
            steps (int, optional): Number of steps per epoch. If None, uses all data.
                                  
        Returns:
            dict: Training history metrics
        """
        # Set max_epochs from passed parameter if provided
        if epochs is not None:
            self.max_epochs = epochs
            logger.info(f"Using provided epochs value: {self.max_epochs}")
        
        # Initialize best metrics to track improvement
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
        # Record initial hyperparameters
        initial_hyperparams = self.cnn_trainer.get_current_hyperparameters()
        
        # Log initial hyperparameters
        logger.info(f"Initial hyperparameters: {initial_hyperparams}")
        self._log_hyperparameters("initial", 0, initial_hyperparams)
        
        # Intervention history
        intervention_history = []
        
        for epoch in range(1, self.max_epochs + 1):
            # First train for this epoch
            train_metrics = self.cnn_trainer.train_epoch()
            
            # Then evaluate on validation set
            val_metrics = self.cnn_trainer.evaluate()
            
            # Convert metrics to float before logging
            try:
                train_loss = float(train_metrics['loss'])
                train_acc = float(train_metrics['accuracy'])
                val_loss = float(val_metrics['loss'])
                val_acc = float(val_metrics['accuracy'])
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting metrics to float: {e}")
                train_loss = train_metrics['loss']
                train_acc = train_metrics['accuracy']
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']
            
            # Log to console
            logger.info(f"Epoch {epoch}/{self.max_epochs} - "
                       f"train_loss: {train_loss:.4f}, "
                       f"train_acc: {train_acc:.4f}, "
                       f"val_loss: {val_loss:.4f}, "
                       f"val_acc: {val_acc:.4f}")
            
            # Update best metrics
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                # Additional logic for saving best model could go here
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            # Check if RL intervention should occur
            if self._should_intervene(epoch):
                # Get current hyperparameters
                current_hyperparams = self.cnn_trainer.get_current_hyperparameters()
                
                # Log before intervention
                logger.info(f"RL INTERVENTION at epoch {epoch}")
                logger.info(f"Current hyperparameters before intervention: {current_hyperparams}")
                
                # Get RL agent action
                state = self._get_state_for_agent()
                action = self.rl_agent.get_action(state)
                
                # Log proposed action
                logger.info(f"RL agent action: {action}")
                
                # Apply the hyperparameter changes
                new_hyperparams = self.cnn_trainer.apply_hyperparameter_action(action)
                
                # Create detailed comparison log
                comparison = {}
                for param_name in set(list(current_hyperparams.keys()) + list(new_hyperparams.keys())):
                    old_val = current_hyperparams.get(param_name, "N/A")
                    new_val = new_hyperparams.get(param_name, "N/A")
                    
                    if old_val != "N/A" and new_val != "N/A":
                        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                            change = new_val - old_val
                            pct_change = (change / old_val) * 100 if old_val != 0 else float('inf')
                            comparison[param_name] = {
                                "before": old_val,
                                "after": new_val,
                                "absolute_change": change,
                                "percent_change": pct_change
                            }
                        else:
                            comparison[param_name] = {
                                "before": str(old_val),
                                "after": str(new_val)
                            }
                
                # Log the comparison
                logger.info("Hyperparameter changes:")
                for param, changes in comparison.items():
                    if "percent_change" in changes:
                        logger.info(f"  {param}: {changes['before']:.6f} → {changes['after']:.6f} ({changes['percent_change']:+.2f}%)")
                    else:
                        logger.info(f"  {param}: {changes['before']} → {changes['after']}")
                
                # Record intervention
                intervention_record = {
                    "epoch": epoch,
                    "hyperparameters_before": current_hyperparams,
                    "action": action,
                    "hyperparameters_after": new_hyperparams,
                    "comparison": comparison,
                    "timestamp": datetime.now().isoformat()
                }
                intervention_history.append(intervention_record)
                
                # Log to file
                self._log_hyperparameters(f"intervention_{len(intervention_history)}", epoch, 
                                          new_hyperparams, current_hyperparams, comparison)
                
                # Log to wandb if enabled
                if self.use_wandb:
                    wandb.log({
                        "rl/intervention_epoch": epoch,
                        "rl/intervention_count": len(intervention_history),
                        **{f"rl/before_{k}": v for k, v in current_hyperparams.items()},
                        **{f"rl/after_{k}": v for k, v in new_hyperparams.items()},
                        **{f"rl/change_{k}": changes["percent_change"] 
                           for k, changes in comparison.items() if "percent_change" in changes}
                    })
            
            # ...rest of the epoch code...
        
        # Save intervention history
        self._save_intervention_history(intervention_history)
        
        # ...rest of training code...
        
        return self.history  # Changed from self.metrics_history to self.history
    
    def _log_hyperparameters(self, event_type, epoch, hyperparams, previous_hyperparams=None, comparison=None):
        """Log hyperparameters to JSON file"""
        log_entry = {
            "event_type": event_type,
            "epoch": epoch,
            "hyperparameters": hyperparams,
            "timestamp": datetime.now().isoformat()
        }
        
        if previous_hyperparams:
            log_entry["previous_hyperparameters"] = previous_hyperparams
            
        if comparison:
            log_entry["comparison"] = comparison
            
        try:
            # Append to log file
            with open(self.hp_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to hyperparameter log file: {e}")
    
    def _save_intervention_history(self, intervention_history):
        """Save full intervention history to JSON file"""
        try:
            history_file = os.path.join(self.hp_log_dir, "intervention_history.json")
            with open(history_file, 'w') as f:
                json.dump(intervention_history, f, indent=2)
            logger.info(f"Saved intervention history to {history_file}")
        except Exception as e:
            logger.error(f"Error saving intervention history: {e}")
