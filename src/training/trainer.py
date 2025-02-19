import torch
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gymnasium as gym

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = model.get_optimizer()

    def train(self, epochs=10):
        wandb.init(project="eye-disease-cnn")
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_acc, metrics = self.evaluate()
            wandb.log({
                'train_loss': train_loss / len(self.train_loader),
                'val_accuracy': val_acc,
                'val_precision': metrics['precision'],
                'val_recall': metrics['recall'],
                'val_f1': metrics['f1']
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
        return best_val_acc

    def evaluate(self):
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1).cpu().numpy()
                predictions.extend(pred)
                targets.extend(target.numpy())
        
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        return accuracy, {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
