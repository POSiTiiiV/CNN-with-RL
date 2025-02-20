import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.panel import Panel
import sys
from torch.amp import autocast, GradScaler

console = Console()

def create_progress():
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
        console=console,
        expand=True,
        transient=True,  # Allow progress bar to be replaced
        refresh_per_second=4
    )

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
        self.progress = None
    
    def log_info(self, message, title=None):
        if self.progress:
            self.progress.stop()
        console.print(Panel.fit(message, title=title))
        if self.progress:
            self.progress.start()
        
    def train(self, epochs=1):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_history = []
        
        self.progress = create_progress()
        self.progress.start()
        
        self.log_info("[bold blue]Starting training...", title="Training Phase")
        
        train_task = self.progress.add_task(
            "[cyan]Training",
            total=len(self.train_loader) * epochs,
            visible=True
        )
        
        try:
            self.optimizer.zero_grad()
            for epoch in range(epochs):
                for batch_idx, batch in enumerate(self.train_loader):
                    if isinstance(batch, (list, tuple)):
                        data, target = batch
                    else:
                        raise ValueError("Batch must contain both data and target")
                    
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    if batch_idx == 0 and epoch == 0:
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
                        batch_metrics = {
                            'batch': batch_idx,
                            'loss': loss.item() * self.gradient_accumulation_steps,
                            'accuracy': batch_accuracy,
                            'total_samples': total
                        }
                        epoch_history.append(batch_metrics)
                        
                        self.progress.update(
                            train_task,
                            advance=1,
                            description=f"[cyan]Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, Acc: {batch_accuracy:.2f}%"
                        )
                        
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
            
            if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            print("\nStarting validation...")
            val_metrics = self.validate()
            
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
            
            self.progress.stop()
            return val_metrics['accuracy']
            
        except Exception as e:
            if self.progress:
                self.progress.stop()
            raise e
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        if self.progress:
            self.progress.stop()
        
        self.progress = create_progress()
        self.progress.start()
        
        self.log_info("[bold green]Starting validation...", title="Validation Phase")
        
        val_task = self.progress.add_task(
            "[green]Validating",
            total=len(self.val_loader),
            visible=True
        )
        
        try:
            with torch.no_grad(), autocast('cuda'):
                for batch_idx, batch in enumerate(self.val_loader):
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
                        
                        self.progress.update(
                            val_task,
                            advance=1,
                            description=f"[green]Validating - Loss: {loss.item():.4f}, Acc: {current_accuracy:.2f}%"
                        )
                        
                    except Exception as e:
                        print(f"\nError in validation batch {batch_idx}:")
                        print(f"Input shape: {data.shape}")
                        print(f"Target shape: {target.shape}")
                        raise e
            
            self.progress.stop()
            
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
            
            console.print(Panel.fit(
                f"[bold]Validation Results\n"
                f"Total samples: {total}\n"
                f"Correct predictions: {correct}\n"
                f"Accuracy: {accuracy:.2f}%\n"
                f"Average Loss: {avg_loss:.4f}",
                title="Results",
                border_style="green"
            ))
            
            return metrics
            
        except Exception as e:
            if self.progress:
                self.progress.stop()
            raise e
