import os
import time
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CNNTrainer:
    """
    Trainer class for CNN models.
    Manages the training process, evaluation, and hyperparameter updates.
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        Initialize the trainer with model and datasets.
        
        Args:
            model: CNN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Parse configuration
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.optimizer_type = config.get('optimizer_type', 'adam')
        
        # Initialize loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
        
        # Initialize optimizer
        self.optimizer = self.get_optimizer()
        
        # Initialize metrics tracking
        self.current_val_loss = float('inf')
        self.current_val_accuracy = 0.0
        self.current_train_loss = float('inf')
        self.current_train_accuracy = 0.0
        
        # Get device
        self.device = next(model.parameters()).device
        
    def get_optimizer(self):
        """
        Create optimizer based on hyperparameters.
        
        Returns:
            torch.optim.Optimizer: The optimizer
        """
        optimizer_type = self.model.hyperparams.get('optimizer_type', self.optimizer_type)
        learning_rate = self.model.hyperparams.get('learning_rate', self.learning_rate)
        weight_decay = self.model.hyperparams.get('weight_decay', self.weight_decay)
        
        if optimizer_type == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.warning(f"Unknown optimizer type: {optimizer_type}, using Adam")
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def train_epoch(self):
        """
        Train model for one epoch.
        
        Returns:
            dict: Dictionary containing 'loss' and 'accuracy' for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        with tqdm(total=len(self.train_loader), desc="Training") as pbar:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                
                # Handle different prediction scenarios
                if targets.dim() > 1 and targets.size(1) > 1:  # One-hot encoded targets
                    # For one-hot encoded targets, get class indices
                    _, predicted_indices = torch.max(outputs.data, 1)
                    _, target_indices = torch.max(targets.data, 1)
                    total += targets.size(0)
                    correct += (predicted_indices == target_indices).sum().item()
                else:  # Class indices
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Update current metrics
        self.current_train_loss = avg_loss
        self.current_train_accuracy = accuracy / 100.0  # Store as decimal
        
        # Return dictionary with metrics (standardized format)
        return {
            'loss': avg_loss,
            'accuracy': accuracy / 100.0  # Return as decimal
        }
    
    def _get_num_classes(self):
        """Get the number of output classes in a model-agnostic way"""
        # Try different common attribute names
        if hasattr(self.model, 'num_classes'):
            return self.model.num_classes
        
        # Look for the final layer based on common naming conventions
        for attr_name in ['fc', 'classifier', 'linear', 'out_layer']:
            if hasattr(self.model, attr_name):
                layer = getattr(self.model, attr_name)
                if hasattr(layer, 'out_features'):
                    return layer.out_features
        
        # If we can't find it directly, examine the model's output dimension
        # This requires a sample input
        if hasattr(self, 'train_loader') and self.train_loader:
            try:
                sample_input, _ = next(iter(self.train_loader))
                sample_input = sample_input[:1].to(self.device)  # Take just one sample
                with torch.no_grad():
                    output = self.model(sample_input)
                return output.size(1)  # Output dimension should be [batch_size, num_classes]
            except:
                pass
        
        # Fallback to binary classification as a safe default
        return 2

    def evaluate(self, data_loader=None):
        """
        Evaluate model on validation set.
        
        Returns:
            dict: Dictionary containing 'loss' and 'accuracy' for validation
        """
        if data_loader is None:
            data_loader = self.val_loader
            
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Validation") as pbar:
                for batch_idx, (inputs, targets) in enumerate(data_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, targets)
                    
                    # Update metrics
                    total_loss += loss.item()
                    
                    # Handle different prediction scenarios
                    if targets.dim() > 1 and targets.size(1) > 1:  # One-hot encoded targets
                        # For one-hot encoded targets, get class indices
                        _, predicted_indices = torch.max(outputs.data, 1)
                        _, target_indices = torch.max(targets.data, 1)
                        total += targets.size(0)
                        correct += (predicted_indices == target_indices).sum().item()
                    else:  # Class indices
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': total_loss / (batch_idx + 1),
                        'acc': 100. * correct / total
                    })
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        # Update current metrics
        self.current_val_loss = avg_loss
        self.current_val_accuracy = accuracy / 100.0  # Store as decimal
        
        # Return dictionary with metrics (standardized format)
        return {
            'loss': avg_loss,
            'accuracy': accuracy / 100.0  # Return as decimal
        }
    
    def get_current_val_loss(self):
        """Get current validation loss."""
        return self.current_val_loss
    
    def get_current_val_accuracy(self):
        """Get current validation accuracy."""
        return self.current_val_accuracy
    
    def get_current_train_loss(self):
        """Get current training loss."""
        return self.current_train_loss
    
    def get_current_train_accuracy(self):
        """Get current training accuracy."""
        return self.current_train_accuracy
    
    def get_current_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def update_hyperparameters(self, hyperparams):
        """
        Update model hyperparameters.
        
        Args:
            hyperparams (dict): New hyperparameters
        """
        # Update model hyperparameters
        self.model.update_hyperparams(hyperparams)
        
        # Create new optimizer with updated parameters
        self.optimizer = self.get_optimizer()
        
        logger.info(f"Updated hyperparameters: {hyperparams}")
        logger.info(f"New learning rate: {self.get_current_lr()}")
        
    def get_current_hyperparameters(self):
        """
        Get current hyperparameters of the model and optimizer.
        
        Returns:
            dict: Current hyperparameter values
        """
        params = {
            'learning_rate': self.current_lr,
            'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0.0)
        }
        
        # Add dropout rate if available
        if hasattr(self.model, 'dropout_rate'):
            params['dropout_rate'] = self.model.dropout_rate
            
        return params

    def get_state(self):
        """Get the current state of training for RL agent"""
        return {
            "epoch": self.current_epoch,
            "train_loss": self.current_metrics.get("train_loss", 0),
            "train_acc": self.current_metrics.get("train_acc", 0),
            "val_loss": self.current_metrics.get("val_loss", 0),
            "val_acc": self.current_metrics.get("val_acc", 0),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "momentum": self.optimizer.param_groups[0].get("momentum", 0),
            "epochs_without_improvement": self.epochs_without_improvement,
            "batch_size": self.batch_size
        }

    def apply_hyperparameter_action(self, action):
        """
        Apply hyperparameter changes from RL agent action.
        
        Args:
            action: Dictionary of hyperparameter actions from RL agent
        
        Returns:
            dict: Updated hyperparameters after applying actions
        """
        current_params = self.get_current_hyperparameters()
        logger.info(f"Applying hyperparameter action: {action}")
        logger.info(f"Current hyperparameters: {current_params}")
        
        # Apply learning rate changes
        if 'learning_rate' in action:
            new_lr = action['learning_rate']
            logger.info(f"Changing learning rate: {self.current_lr:.6f} → {new_lr:.6f}")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
            
        # Apply weight decay changes
        if 'weight_decay' in action:
            new_wd = action['weight_decay']
            current_wd = self.optimizer.param_groups[0].get('weight_decay', 0.0)
            logger.info(f"Changing weight decay: {current_wd:.6f} → {new_wd:.6f}")
            
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = new_wd
                
        # Apply dropout changes if model supports it
        if 'dropout_rate' in action and hasattr(self.model, 'update_dropout'):
            current_dropout = getattr(self.model, 'dropout_rate', 0.0)
            new_dropout = action['dropout_rate']
            logger.info(f"Changing dropout rate: {current_dropout:.3f} → {new_dropout:.3f}")
            self.model.update_dropout(new_dropout)
            
        # Log the final updated hyperparameters
        updated_params = self.get_current_hyperparameters()
        logger.info(f"Updated hyperparameters: {updated_params}")
        
        return updated_params
        
    def save_model(self, filepath):
        """
        Save the model weights to the specified file path.
        
        Args:
            filepath (str): Path where the model should be saved
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Create a dictionary with all the necessary data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': self.get_current_hyperparameters(),
            'val_loss': self.current_val_loss,
            'val_accuracy': self.current_val_accuracy
        }
        
        # Save the model
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model weights from the specified file path.
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return False
            
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # Load metrics if available
            if 'val_loss' in checkpoint:
                self.current_val_loss = checkpoint['val_loss']
            if 'val_accuracy' in checkpoint:
                self.current_val_accuracy = checkpoint['val_accuracy']
                
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}")
            return False
