import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()
        # Get parameters from config
        in_channels = config.get('in_channels', 3)
        num_classes = config.get('num_classes', 10)
        
        # Default architecture, will be optimized by RL
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming 32x32 input -> 8x8 after two pooling
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def update_hyperparams(self, hyperparams):
        """
        Update model hyperparameters based on RL suggestions
        """
        # Implementation for dynamically updating model architecture
        # Will be expanded based on specific hyperparameters to optimize
        pass
