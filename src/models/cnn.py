import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet34, ResNet34_Weights
from collections import OrderedDict

class PretrainedCNN(nn.Module):
    def __init__(self, config):
        super(PretrainedCNN, self).__init__()
        
        # Extract configuration parameters
        self.num_classes = config.get('num_classes', 10)
        self.fc_layers = config.get('fc_layers', [512, 256])
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer_type = config.get('optimizer_type', 'adam')
        self.freeze_backbone = config.get('freeze_backbone', True)
        
        # Load pretrained ResNet34
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the number of features from the last layer of ResNet
        backbone_output_features = self.backbone.fc.in_features
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Build a new flexible head
        head_layers = OrderedDict()
        
        # First layer connects to backbone output
        prev_size = backbone_output_features
        
        # Add specified hidden layers
        for i, size in enumerate(self.fc_layers):
            head_layers[f'fc{i+1}'] = nn.Linear(prev_size, size)
            head_layers[f'relu{i+1}'] = nn.ReLU(inplace=True)
            head_layers[f'dropout{i+1}'] = nn.Dropout(self.dropout_rate)
            prev_size = size
        
        # Add final classification layer
        head_layers['output'] = nn.Linear(prev_size, self.num_classes)
        
        # Create sequential module for the head
        self.head = nn.Sequential(head_layers)
        
        # Store hyperparameters
        self.hyperparams = {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'fc_layers': self.fc_layers,
            'optimizer_type': self.optimizer_type,
        }

    def forward(self, x):
        """Forward pass through the network"""
        # Pass through backbone
        x = self.backbone(x)
        # Flatten the output
        x = torch.flatten(x, 1)
        # Pass through custom head
        x = self.head(x)
        return x

    def update_hyperparams(self, hyperparams):
        """
        Update model hyperparameters dynamically
        
        Args:
            hyperparams: Dictionary containing hyperparameters to update
        """
        # Log current hyperparameters before update
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Current hyperparameters before RL agent intervention:")
        for param_name, param_value in self.hyperparams.items():
            logger.info(f"  {param_name}: {param_value}")
        
        # Update stored hyperparameter values
        for param, value in hyperparams.items():
            if param in self.hyperparams:
                self.hyperparams[param] = value
        
        # Log updated hyperparameters
        logger.info("Updated hyperparameters after RL agent intervention:")
        for param_name, param_value in self.hyperparams.items():
            logger.info(f"  {param_name}: {param_value}")
            
        # Handle dropout rate changes
        if 'dropout_rate' in hyperparams:
            new_rate = hyperparams['dropout_rate']
            # Update dropout layers in the head
            for name, module in self.head.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = new_rate
        
        # Handle changes to fully connected layer sizes
        if 'fc_layers' in hyperparams:
            new_layers = hyperparams['fc_layers']
            
            # Only rebuild if the layer sizes actually changed
            if new_layers != self.fc_layers:
                self.fc_layers = new_layers
                
                # Get backbone output size - fixed to use the correct way to determine ResNet output features
                # ResNet34 always has 512 features at the end of the backbone
                backbone_output_features = 512  # Fixed output size for ResNet34
                
                # Build new head
                head_layers = OrderedDict()
                
                # First layer connects to backbone output
                prev_size = backbone_output_features
                
                # Add specified hidden layers
                for i, size in enumerate(self.fc_layers):
                    head_layers[f'fc{i+1}'] = nn.Linear(prev_size, size)
                    head_layers[f'relu{i+1}'] = nn.ReLU(inplace=True)
                    head_layers[f'dropout{i+1}'] = nn.Dropout(self.hyperparams['dropout_rate'])
                    prev_size = size
                
                # Add final classification layer
                head_layers['output'] = nn.Linear(prev_size, self.num_classes)
                
                # Create sequential module for the head
                self.head = nn.Sequential(head_layers)
                
                # Move to correct device if model is already on a device
                if next(self.parameters()).is_cuda:
                    device = next(self.parameters()).device
                    self.head = self.head.to(device)

    def get_optimizer(self):
        """
        Get optimizer based on current hyperparameters
        
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        # Determine which parameters to optimize
        if self.freeze_backbone:
            params = self.head.parameters()
        else:
            params = self.parameters()
        
        # Create optimizer based on type
        if self.hyperparams['optimizer_type'].lower() == 'adam':
            optimizer = optim.Adam(
                params,
                lr=self.hyperparams['learning_rate'],
                weight_decay=self.hyperparams['weight_decay']
            )
        elif self.hyperparams['optimizer_type'].lower() == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=self.hyperparams['learning_rate'],
                momentum=0.9,
                weight_decay=self.hyperparams['weight_decay']
            )
        elif self.hyperparams['optimizer_type'].lower() == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=self.hyperparams['learning_rate'],
                weight_decay=self.hyperparams['weight_decay']
            )
        else:
            # Default to Adam
            optimizer = optim.Adam(
                params,
                lr=self.hyperparams['learning_rate'],
                weight_decay=self.hyperparams['weight_decay']
            )
            
        return optimizer
        
    def configure_for_mixed_precision(self):
        """
        Configure model for mixed precision training
        
        Returns:
            Tuple containing model and optimizer for mixed precision training
        """
        from torch.cuda.amp import GradScaler, autocast
        
        # Create gradient scaler for mixed precision training
        scaler = GradScaler()
        
        # Get optimizer
        optimizer = self.get_optimizer()
        
        return self, optimizer, scaler, autocast
