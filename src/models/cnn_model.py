import torch
import torch.nn as nn
from torchvision import models

class FlexibleCNN(nn.Module):
    def __init__(self, num_classes, base_model='resnet34', hyperparams=None):
        super(FlexibleCNN, self).__init__()
        self.hyperparams = hyperparams or self._default_hyperparams()
        
        # Load pretrained base model
        if base_model == 'resnet34':
            self.base_model = models.resnet34(pretrained=True)
            feature_dim = self.base_model.fc.in_features  # ResNet34 has 512 features vs ResNet50's 2048
            self.base_model.fc = nn.Identity()
        
        # Custom CNN layers
        layers = []
        current_dim = feature_dim
        
        for layer_size in self.hyperparams['layer_sizes']:
            layers.extend([
                nn.Linear(current_dim, layer_size),
                nn.ReLU(),
                nn.Dropout(self.hyperparams['dropout_rate'])
            ])
            current_dim = layer_size
        
        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def _default_hyperparams(self):
        return {
            'layer_sizes': [512, 256],  # Adjusted for ResNet34's smaller feature space
            'dropout_rate': 0.5,
            'learning_rate': 0.001
        }

    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hyperparams['learning_rate']
        )
