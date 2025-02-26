import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedCNN(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=2, pretrained=True):
        super(PretrainedCNN, self).__init__()
        self.num_classes = num_classes
        
        # Load the pretrained model
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            # Modify the final fully connected layer
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, num_classes)
            self.model = base_model
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, num_classes)
            self.model = base_model
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, num_classes)
            self.model = base_model
        elif model_name == 'vgg16':
            base_model = models.vgg16(pretrained=pretrained)
            in_features = base_model.classifier[6].in_features
            base_model.classifier[6] = nn.Linear(in_features, num_classes)
            self.model = base_model
        # Add more model options as needed
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)
