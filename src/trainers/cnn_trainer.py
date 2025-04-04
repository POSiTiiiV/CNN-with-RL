import torch
import logging
from tqdm import tqdm
from io import StringIO
from ..utils.utils import get_optimizer

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
        self.use_mixed_precision = config.get('use_mixed_precision', None)
        
        # Automatically enable mixed precision for GPU training if not explicitly set
        if torch.cuda.is_available() and self.use_mixed_precision is None:
            self.use_mixed_precision = True
            logger.info("Automatically enabling mixed precision for GPU training")
        
        # Initialize mixed precision training if enabled
        self.scaler = torch.amp.GradScaler() if self.use_mixed_precision else None
        
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
        
        # Check GPU memory and adjust batch size if needed
        self._check_and_adjust_batch_size()
        
        # Log initial hyperparameters
        logger.info("Initializing CNNTrainer with hyperparameters:")
        
        # Replace Rich Table with plain logging
        print('\n')
        logger.info("Model Hyperparameters:")
        print('\n')
        for param_name, param_value in self.model.hyperparams.items():
            if isinstance(param_value, float):
                logger.info(f"{param_name}: {param_value:.6f}")
            else:
                logger.info(f"{param_name}: {param_value}")

        if self.use_mixed_precision:
            logger.info("Using mixed precision training for better GPU utilization")
            
    def _check_and_adjust_batch_size(self):
        """
        Check available GPU memory and adjust batch size if necessary to avoid OOM errors.
        """
        if torch.cuda.is_available():
            try:
                # Get GPU memory information
                device = torch.cuda.current_device()
                gpu_properties = torch.cuda.get_device_properties(device)
                total_memory = gpu_properties.total_memory / 1024**2  # Convert to MB
                
                # Get current GPU memory usage
                allocated_memory = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
                cached_memory = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
                
                # Calculate free memory (approximate)
                free_memory = total_memory - allocated_memory - cached_memory
                
                logger.info(f"GPU Memory: Total={total_memory:.0f}MB, Free={free_memory:.0f}MB, Used={allocated_memory:.0f}MB")
                
                # If free memory is low and batch size is high, reduce batch size
                if free_memory < 2000 and self.batch_size > 8:  # Threshold of 2GB and batch size > 8
                    original_batch_size = self.batch_size
                    # Reduce batch size based on available memory
                    if free_memory < 1000:
                        self.batch_size = max(1, self.batch_size // 4)  # Aggressive reduction
                    elif free_memory < 1500:
                        self.batch_size = max(2, self.batch_size // 2)  # Moderate reduction
                    else:
                        self.batch_size = max(4, int(self.batch_size * 0.75))  # Mild reduction
                    
                    # Update model hyperparams
                    self.model.hyperparams['batch_size'] = self.batch_size
                    
                    logger.warning(f"⚠️ Reduced batch size from {original_batch_size} to {self.batch_size} due to limited GPU memory")
                    
                    # Force CUDA cache clear
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"Warning: Unable to check GPU memory: {str(e)}")
                # Continue with original batch size
    
    def get_optimizer(self):
        """
        Create optimizer based on hyperparameters.
        
        Returns:
            torch.optim.Optimizer: The optimizer
        """
        return get_optimizer(
            self.model,
            optimizer_type=self.model.hyperparams.get('optimizer_type', self.optimizer_type),
            learning_rate=self.model.hyperparams.get('learning_rate', self.learning_rate),
            weight_decay=self.model.hyperparams.get('weight_decay', self.weight_decay)
        )
    
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
        nan_detected = False
        max_grad_norm = 1.0  # Gradient clipping threshold
        
        # Use tqdm progress bar instead of Rich
        with tqdm(total=len(self.train_loader), desc="Training", unit="batch") as progress:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                # Try to clear GPU cache if available
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Mixed precision training path
                if self.use_mixed_precision:
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        # Forward pass with mixed precision
                        outputs = self.model(inputs)
                        
                        # Check for NaN in outputs
                        if torch.isnan(outputs).any():
                            logger.warning(f"Warning: NaN detected in model outputs during training at batch {batch_idx}")
                            nan_detected = True
                            # Replace NaN values with zeros to continue computation
                            outputs = torch.nan_to_num(outputs, nan=0.0)
                            
                        loss = self.criterion(outputs, targets)
                        
                        # Check for NaN in loss
                        if torch.isnan(loss).any():
                            logger.warning(f"Warning: NaN detected in loss during training at batch {batch_idx}")
                            nan_detected = True
                            # Skip this batch if loss is NaN
                            continue
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Clip gradients to prevent explosion
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Skip step if NaN gradients are detected
                    skip_step = False
                    for param in self.model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            logger.warning(f"Warning: NaN detected in gradients during training at batch {batch_idx}")
                            nan_detected = True
                            skip_step = True
                            break
                    
                    if not skip_step:
                        self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                # Standard precision training path
                else:
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Check for NaN in outputs
                    if torch.isnan(outputs).any():
                        logger.warning(f"Warning: NaN detected in model outputs during training at batch {batch_idx}")
                        nan_detected = True
                        # Replace NaN values with zeros to continue computation
                        outputs = torch.nan_to_num(outputs, nan=0.0)
                        
                    loss = self.criterion(outputs, targets)
                    
                    # Check for NaN in loss
                    if torch.isnan(loss).any():
                        logger.warning(f"Warning: NaN detected in loss during training at batch {batch_idx}")
                        nan_detected = True
                        # Skip this batch if loss is NaN
                        continue
                    
                    # Backward pass
                    loss.backward()
                    
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Skip step if NaN gradients are detected
                    skip_step = False
                    for param in self.model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            logger.warning(f"Warning: NaN detected in gradients during training at batch {batch_idx}")
                            nan_detected = True
                            skip_step = True
                            break
                    
                    if not skip_step:
                        self.optimizer.step()
                
                # Update metrics (only if we didn't skip due to NaNs)
                if not torch.isnan(loss).any():
                    total_loss += loss.item()
                
                # Handle different prediction scenarios (using less GPU memory)
                with torch.no_grad():  # Ensure no gradients are tracked during prediction comparison
                    if targets.dim() > 1 and targets.size(1) > 1:  # One-hot encoded targets
                        # For one-hot encoded targets, get class indices
                        _, predicted_indices = torch.max(outputs.data, 1)
                        _, target_indices = torch.max(targets.data, 1)
                        batch_correct = (predicted_indices == target_indices).sum().item()
                    else:  # Class indices
                        _, predicted = torch.max(outputs.data, 1)
                        batch_correct = (predicted == targets).sum().item()
                    
                    # Update counters
                    total += targets.size(0)
                    correct += batch_correct
                
                # Free memory explicitly
                del inputs, targets, outputs, loss
                if 'predicted' in locals():
                    del predicted
                if 'predicted_indices' in locals():
                    del predicted_indices, target_indices
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0
                accuracy = 100. * correct / total if total > 0 else 0
                progress.set_postfix(loss=avg_loss, acc=accuracy)
                progress.update(1)
        
        # Calculate epoch metrics
        if len(self.train_loader) > 0 and not nan_detected:
            avg_loss = total_loss / len(self.train_loader)
        else:
            avg_loss = float('nan')
            logger.warning("Warning: NaN detected in training metrics")
            
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        # Update current metrics
        self.current_train_loss = avg_loss
        self.current_train_accuracy = accuracy / 100.0  # Store as decimal
        
        # Check if training has gone completely off the rails
        if nan_detected or torch.isnan(torch.tensor(avg_loss)):
            logger.error("Training has become unstable with NaN values - the RL agent will be signaled to reduce the learning rate or adjust regularization")
        
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
        nan_detected = False
        
        with torch.no_grad():
            # Use tqdm progress bar instead of Rich
            with tqdm(total=len(data_loader), desc="Validation", unit="batch") as progress:
                for batch_idx, (inputs, targets) in enumerate(data_loader):
                    # Try to clear GPU cache if available
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Check for NaN values in the outputs
                    if torch.isnan(outputs).any():
                        logger.warning(f"Warning: NaN detected in model outputs during evaluation at batch {batch_idx}")
                        nan_detected = True
                        # Replace NaN with zeros for computation to continue
                        outputs = torch.nan_to_num(outputs, nan=0.0)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, targets)
                    
                    # Check for NaN in loss
                    if torch.isnan(loss).any():
                        logger.warning(f"Warning: NaN detected in loss during evaluation at batch {batch_idx}")
                        nan_detected = True
                        # Use a small positive value instead of NaN for the batch loss
                        batch_loss = 1.0
                    else:
                        batch_loss = loss.item()
                    
                    # Update metrics
                    total_loss += batch_loss
                    
                    # Handle different prediction scenarios
                    if targets.dim() > 1 and targets.size(1) > 1:  # One-hot encoded targets
                        # For one-hot encoded targets, get class indices
                        _, predicted_indices = torch.max(outputs.data, 1)
                        _, target_indices = torch.max(targets.data, 1)
                        batch_correct = (predicted_indices == target_indices).sum().item()
                    else:  # Class indices
                        _, predicted = torch.max(outputs.data, 1)
                        batch_correct = (predicted == targets).sum().item()
                    
                    # Update counters
                    total += targets.size(0)
                    correct += batch_correct
                    
                    # Free memory explicitly
                    del inputs, targets, outputs, loss
                    if 'predicted' in locals():
                        del predicted
                    if 'predicted_indices' in locals():
                        del predicted_indices, target_indices
                    
                    # Update progress bar
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100. * correct / total if total > 0 else 0
                    progress.set_postfix(loss=avg_loss, acc=accuracy)
                    progress.update(1)
        
        # Calculate metrics
        if len(data_loader) > 0 and not nan_detected:
            avg_loss = total_loss / len(data_loader)
        else:
            avg_loss = float('nan')
            logger.warning("Warning: NaN detected in evaluation metrics")
            
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        # Update current metrics
        self.current_val_loss = avg_loss
        self.current_val_accuracy = accuracy / 100.0  # Store as decimal
        
        # Check if evaluation has gone completely off the rails
        if nan_detected or torch.isnan(torch.tensor(avg_loss)):
            logger.error("Model evaluation has unstable NaN values - the RL agent will be notified to adjust hyperparameters")
        
        # Return dictionary with metrics (standardized format)
        return {
            'loss': avg_loss,
            'accuracy': accuracy / 100.0  # Return as decimal
        }
