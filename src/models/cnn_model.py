import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(self, num_classes, base_model='resnet34', hyperparams=None):
        super(FlexibleCNN, self).__init__()
        self.hyperparams = hyperparams or self._default_hyperparams()
        
        # Load pretrained base model with efficient memory loading
        if base_model == 'resnet34':
            self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            
            # Freeze early layers to speed up training
            for param in list(self.base_model.parameters())[:-9]:  # Keep last 3 layers trainable
                param.requires_grad = False
        
        # Custom CNN layers with efficient memory usage
        self.fc1 = nn.Linear(feature_dim, self.hyperparams['layer_sizes'][0])
        self.dropout = nn.Dropout(self.hyperparams['dropout_rate'])
        self.final = nn.Linear(self.hyperparams['layer_sizes'][0], num_classes)
        
        # Initialize weights properly
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.final.weight)

    def _default_hyperparams(self):
        return {
            'layer_sizes': [512],  # Simplified architecture
            'dropout_rate': 0.5,
            'learning_rate': 0.001
        }

    def forward(self, x):
        # Efficient forward pass
        with torch.cuda.amp.autocast():
            features = self.base_model(x)
            x = F.relu(self.fc1(features), inplace=True)  # inplace ReLU saves memory
            x = self.dropout(x)
            x = self.final(x)
        return x

    def get_optimizer(self):
        # Use different learning rates for frozen and trainable layers
        base_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'base_model' in name:
                    base_params.append(param)
                else:
                    classifier_params.append(param)
        
        return torch.optim.Adam([
            {'params': base_params, 'lr': self.hyperparams['learning_rate'] * 0.1},
            {'params': classifier_params, 'lr': self.hyperparams['learning_rate']}
        ])
