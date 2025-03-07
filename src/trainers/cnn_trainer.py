import torch
import logging
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
 