import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gymnasium as gym
import sys
from torch.cuda.amp import autocast, GradScaler

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, use_wandb=False, gradient_accumulation_steps=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = model.get_optimizer()
        self.use_wandb = use_wandb
        self.scaler = GradScaler()  # For mixed precision training
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
    def train(self, epochs=1):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_history = []
        
        print(f"\nStarting training for {epochs} epochs...")
        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            total=len(self.train_loader),
            file=sys.stdout,
            leave=True,
            dynamic_ncols=True
        )
        
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(progress_bar):
            if isinstance(batch, (list, tuple)):
                data, target = batch
            else:
                raise ValueError("Batch must contain both data and target")
            
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            if batch_idx == 0:
                print(f"\nInput shape: {data.shape}")
                print(f"Target shape: {target.shape}")
            
            try:
                # Mixed precision training
                with autocast():
                    output = self.model(data)
                    if batch_idx == 0:
                        print(f"Output shape: {output.shape}")
                    
                    target_class = torch.argmax(target, dim=1)
                    loss = self.criterion(output, target_class)
                    loss = loss / self.gradient_accumulation_steps  # Scale loss for accumulation
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                with torch.no_grad():
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target_class).sum().item()
                
                # Calculate metrics for this batch
                batch_accuracy = 100. * correct / total
                batch_metrics = {
                    'batch': batch_idx,
                    'loss': loss.item() * self.gradient_accumulation_steps,
                    'accuracy': batch_accuracy,
                    'total_samples': total
                }
                epoch_history.append(batch_metrics)
                
                # Update progress bar with current metrics
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'acc': f'{batch_accuracy:.2f}%'
                })
                
                if batch_idx % 10 == 0 and self.use_wandb:
                    wandb.log({
                        "batch": batch_idx,
                        "train_loss": loss.item() * self.gradient_accumulation_steps,
                        "train_accuracy": batch_accuracy,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                        
            except Exception as e:
                print(f"\nError in batch {batch_idx}:")
                print(f"Input shape: {data.shape}")
                print(f"Target shape: {target.shape}")
                raise e
        
        progress_bar.close()
        
        # Handle any remaining gradients
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # Validation
        print("\nStarting validation...")
        val_metrics = self.validate()
        
        # Log epoch summary
        epoch_avg_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        
        if self.use_wandb:
            wandb.log({
                "epoch_avg_loss": epoch_avg_loss,
                "epoch_accuracy": epoch_accuracy,
                "epoch_val_accuracy": val_metrics['accuracy'],
                "epoch_val_loss": val_metrics['loss'],
                "epoch_samples": total
            })
        
        return val_metrics['accuracy']
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        print("\nRunning validation...")
        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            total=len(self.val_loader),
            file=sys.stdout,
            leave=True,
            dynamic_ncols=True
        )
        
        with torch.no_grad(), autocast():
            for batch_idx, batch in enumerate(progress_bar):
                if isinstance(batch, (list, tuple)):
                    data, target = batch
                else:
                    raise ValueError("Batch must contain both data and target")
                
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                if batch_idx == 0:
                    print(f"\nValidation input shape: {data.shape}")
                    print(f"Validation target shape: {target.shape}")
                
                try:
                    output = self.model(data)
                    target_class = torch.argmax(target, dim=1)
                    loss = self.criterion(output, target_class)
                    total_loss += loss.item()
                    
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target_class).sum().item()
                    
                    # Update progress bar
                    current_accuracy = 100. * correct / total
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{current_accuracy:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}:")
                    print(f"Input shape: {data.shape}")
                    print(f"Target shape: {target.shape}")
                    raise e
        
        progress_bar.close()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
        
        if self.use_wandb:
            wandb.log({
                "val_accuracy": accuracy,
                "val_loss": avg_loss,
                "val_correct": correct,
                "val_total": total
            })
        
        print(f"\nValidation Results:")
        print(f"Total samples: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Loss: {avg_loss:.4f}")
        
        return metrics
