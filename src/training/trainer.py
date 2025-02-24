import torch
import torch.nn as nn
import wandb
import numpy as np  # Add numpy import
from rich.console import Console
from rich.panel import Panel
from torch.amp import autocast, GradScaler
from tqdm import tqdm

console = Console()

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, use_wandb=False, gradient_accumulation_steps=4, rl_agent=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = model.get_optimizer()
        self.use_wandb = use_wandb
        self.scaler = GradScaler()  # For mixed precision training
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.rl_agent = rl_agent  # RL agent for adaptive hyperparameter tuning
        self.prev_val_loss = float('inf')
        self.stagnation_count = 0
        self.training_history = []
        self.min_epochs = 5
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def log_info(self, message, title=None):
        console.print(Panel.fit(message, title=title))
        
    def train(self):  # Remove max_epochs parameter
        self.model.train()
        if self.current_epoch == 0:
            best_accuracy = 0
            early_stop = False
            training_cost = 0
        else:
            best_accuracy = max([h['val_acc'] for h in self.training_history]) if self.training_history else 0
            early_stop = False
            training_cost = max(0, self.current_epoch - self.min_epochs)
        
        self.log_info("[bold blue]Continuing training from epoch {}...".format(self.current_epoch), 
                     title="Training Phase")
        
        try:
            # Train for one epoch at a time
            epoch_metrics = self._train_single_epoch()
            val_metrics = self.validate()
            
            # Store training history
            current_metrics = {
                'epoch': self.current_epoch,
                'train_loss': epoch_metrics['loss'],
                'train_acc': epoch_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy']
            }
            self.training_history.append(current_metrics)
            
            # Update best metrics
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            training_cost = max(0, self.current_epoch - self.min_epochs)
            self.current_epoch += 1
            
            if self.use_wandb:
                wandb.log({
                    "epoch": self.current_epoch,
                    "train_loss": epoch_metrics['loss'],
                    "train_accuracy": epoch_metrics['accuracy'],
                    "val_accuracy": val_metrics['accuracy'],
                    "val_loss": val_metrics['loss'],
                    "training_cost": training_cost
                })
            
            # Calculate reward
            improvement_ratio = best_accuracy / max(1, self.current_epoch - self.min_epochs)
            final_reward = improvement_ratio - (training_cost * 0.1)
            
            return {
                'accuracy': best_accuracy,
                'epochs_trained': self.current_epoch,
                'training_cost': training_cost,
                'reward': final_reward
            }
            
        except Exception as e:
            raise e

    def apply_hyperparameter_update(self, action):
        """Apply new hyperparameters from RL agent"""
        new_hyperparams = {
            'learning_rate': float(action[0]),
            'layer_sizes': [int(action[1])],
            'dropout_rate': float(action[2]),
            'weight_decay': 0.01
        }
        
        self.model.hyperparams = new_hyperparams
        self.optimizer = self.model.get_optimizer()
        self.log_info(f"[bold yellow]Updated hyperparameters: {new_hyperparams}", 
                     title="RL Intervention")

    def update_hyperparameters(self, new_params):
        """Update hyperparameters without reinitializing the model"""
        # Convert array-like inputs to dictionary
        if isinstance(new_params, (np.ndarray, list, torch.Tensor)):
            new_params = np.asarray(new_params)  # Convert to numpy array for consistency
            new_hyperparams = {
                'learning_rate': float(new_params[0]),
                'layer_sizes': [int(new_params[1])],
                'dropout_rate': float(new_params[2]),
                'weight_decay': float(new_params[3])  # Include weight decay from action
            }
        else:
            new_hyperparams = new_params
            
        # Update model's hyperparameters
        self.model.hyperparams.update(new_hyperparams)
        
        # Update layer sizes if changed
        if 'layer_sizes' in new_hyperparams:
            new_size = new_hyperparams['layer_sizes'][0]
            old_size = self.model.fc1.out_features
            if new_size != old_size:
                in_features = self.model.fc1.in_features
                self.model.fc1 = nn.Linear(in_features, new_size).to(self.device)
                self.model.final = nn.Linear(new_size, self.model.final.out_features).to(self.device)
                nn.init.kaiming_normal_(self.model.fc1.weight)
                nn.init.kaiming_normal_(self.model.final.weight)
                self.optimizer = self.model.get_optimizer()
                return
        
        # Update dropout rate
        if 'dropout_rate' in new_hyperparams:
            self.model.dropout.p = new_hyperparams['dropout_rate']
        
        # Update learning rate and weight decay
        if 'learning_rate' in new_hyperparams or 'weight_decay' in new_hyperparams:
            for param_group in self.optimizer.param_groups:
                if 'learning_rate' in new_hyperparams:
                    if 'base_model' in str(param_group['params'][0]):
                        param_group['lr'] = new_hyperparams['learning_rate'] * 0.1
                    else:
                        param_group['lr'] = new_hyperparams['learning_rate']
                if 'weight_decay' in new_hyperparams:
                    param_group['weight_decay'] = new_hyperparams['weight_decay']
        
        self.log_info(f"[bold yellow]Updated hyperparameters: {new_hyperparams}", 
                     title="Parameter Update")

    def _train_single_epoch(self):
        total_loss = 0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")):
            if isinstance(batch, (list, tuple)):
                data, target = batch
            else:
                raise ValueError("Batch must contain both data and target")
            
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            if batch_idx == 0 and self.current_epoch == 0:
                self.log_info(
                    f"[bold blue]Data Shapes\n"
                    f"Input: {data.shape}\n"
                    f"Target: {target.shape}\n"
                    f"Output: {self.model(data).shape}",
                    title="Model Information"
                )
            
            try:
                with autocast('cuda'):
                    output = self.model(data)
                    
                    target_class = torch.argmax(target, dim=1)
                    loss = self.criterion(output, target_class)
                    loss = loss / self.gradient_accumulation_steps  # Scale loss for accumulation
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                with torch.no_grad():
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target_class).sum().item()
                
                batch_accuracy = 100. * correct / total
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: Loss = {loss.item() * self.gradient_accumulation_steps:.4f}, Accuracy = {batch_accuracy:.2f}%")
                    if self.use_wandb:
                        wandb.log({
                            "batch": batch_idx,
                            "train_loss": loss.item() * self.gradient_accumulation_steps,
                            "train_accuracy": batch_accuracy,
                            "learning_rate": self.optimizer.param_groups[0]['lr']
                        })
                        
            except Exception as e:
                raise e
        
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        epoch_avg_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        
        return {
            'loss': epoch_avg_loss,
            'accuracy': epoch_accuracy,
            'correct': correct,
            'total': total
        }

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        self.log_info("[bold green]Starting validation...", title="Validation Phase")
        
        try:
            with torch.no_grad(), autocast('cuda'):
                for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                    if isinstance(batch, (list, tuple)):
                        data, target = batch
                    else:
                        raise ValueError("Batch must contain both data and target")
                    
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    if batch_idx == 0:
                        self.log_info(
                            f"[bold green]Validation Shapes\n"
                            f"Input: {data.shape}\n"
                            f"Target: {target.shape}",
                            title="Validation Information"
                        )
                    
                    try:
                        output = self.model(data)
                        target_class = torch.argmax(target, dim=1)
                        loss = self.criterion(output, target_class)
                        total_loss += loss.item()
                        
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target_class).sum().item()
                        
                        current_accuracy = 100. * correct / total
                        
                        if batch_idx % 10 == 0:
                            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {current_accuracy:.2f}%")
                        
                    except Exception as e:
                        raise e
            
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
            
            self.log_info(
                f"[bold]Validation Results\n"
                f"Total samples: {total}\n"
                f"Correct predictions: {correct}\n"
                f"Accuracy: {accuracy:.2f}%\n"
                f"Average Loss: {avg_loss:.4f}",
                title="Results"
            )
            
            return metrics
            
        except Exception as e:
            raise e
