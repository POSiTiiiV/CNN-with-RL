import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gymnasium as gym

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, use_wandb=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = model.get_optimizer()
        self.use_wandb = use_wandb
        
    def train(self, epochs=1):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle different batch types
            if isinstance(batch, (list, tuple)):
                data, target = batch
            else:
                raise ValueError("Batch must contain both data and target")
            
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_accuracy": 100. * correct / total
                })
        
        # Validation
        val_accuracy = self.validate()
        return val_accuracy
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle different batch types
                if isinstance(batch, (list, tuple)):
                    data, target = batch
                else:
                    raise ValueError("Batch must contain both data and target")
                
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        if self.use_wandb:
            wandb.log({"val_accuracy": accuracy})
        return accuracy
