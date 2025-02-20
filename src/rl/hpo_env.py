import gymnasium as gym
import numpy as np
from gymnasium import spaces
import wandb
from ..models.cnn_model import FlexibleCNN
from ..training.trainer import ModelTrainer
from rich.console import Console
from rich.panel import Panel

console = Console()

class HPOEnvironment(gym.Env):
    def __init__(self, trainer, train_loader, val_loader, num_classes, experiment_name="HPO-CNN", dtype=np.float32):
        super(HPOEnvironment, self).__init__()
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([1e-4, 64, 0.1], dtype=dtype),  # lr, layer_size, dropout
            high=np.array([1e-1, 2048, 0.9], dtype=dtype),
            dtype=dtype
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),  # Current lr, layer_size, dropout
            dtype=dtype
        )
        
        self.current_hyperparams = None
        self.episode_count = 0
        self.step_count = 0
        
        # Initialize wandb
        wandb.init(
            project="cnn-with-rl",
            name=experiment_name,
            config={
                "num_classes": num_classes,
                "action_space": {
                    "lr_range": [0.0001, 0.1],
                    "layer_size_range": [64, 2048],
                    "dropout_range": [0.1, 0.9]
                }
            }
        )

        console.print(Panel.fit(
            "[bold blue]HPO Environment Initialized\n"
            f"Action Space: {self.action_space}\n"
            f"Observation Space: {self.observation_space}",
            title="Environment Setup"
        ))

    def reset(self, seed=None, options=None):
        console.print(Panel.fit("[bold yellow]Resetting environment...", title="Reset"))
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.step_count = 0
        
        # Reset model with default hyperparameters
        self.current_hyperparams = {
            'learning_rate': 0.001,
            'layer_sizes': [512],
            'dropout_rate': 0.5
        }
        console.print(f"[blue]Initial hyperparameters: {self.current_hyperparams}")
        
        # Reinitialize model with default hyperparameters
        print("Reinitializing model with default parameters...")
        model = FlexibleCNN(
            num_classes=self.num_classes,
            hyperparams=self.current_hyperparams
        ).to(self.trainer.device)
        
        self.trainer.model = model
        self.trainer.optimizer = model.get_optimizer()
        print("Model reinitialized")
        
        return self._get_observation(), {}

    def step(self, action):
        console.print(Panel.fit(
            f"[bold cyan]Episode {self.episode_count}, Step {self.step_count}\n"
            f"Action: {action}",
            title="Environment Step"
        ))
        self.step_count += 1
        print(f"Episode {self.episode_count}, Step {self.step_count}")
        print(f"Action received: {action}")
        
        # Update hyperparameters
        self.current_hyperparams['learning_rate'] = float(action[0])
        self.current_hyperparams['layer_sizes'] = [int(action[1])]
        self.current_hyperparams['dropout_rate'] = float(action[2])
        print(f"New hyperparameters: {self.current_hyperparams}")
        
        # Reinitialize model with new hyperparameters
        print("Reinitializing model with new hyperparameters...")
        model = FlexibleCNN(
            num_classes=self.num_classes,
            hyperparams=self.current_hyperparams
        ).to(self.trainer.device)
        
        self.trainer.model = model
        self.trainer.optimizer = model.get_optimizer()
        
        # Train and evaluate model
        val_accuracy = 0.0
        try:
            print("Training model for one epoch...")
            val_accuracy = self.trainer.train(epochs=1)
            reward = val_accuracy
            
            # Log metrics to wandb
            wandb.log({
                "episode": self.episode_count,
                "step": self.step_count,
                "reward": reward,
                "validation_accuracy": val_accuracy,
                "learning_rate": self.current_hyperparams['learning_rate'],
                "layer_size": self.current_hyperparams['layer_sizes'][0],
                "dropout_rate": self.current_hyperparams['dropout_rate']
            })
            
            print(f"Training complete - Validation accuracy: {val_accuracy:.2f}%")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            reward = 0.0
        
        done = False
        truncated = False
        info = {
            'val_accuracy': val_accuracy,
            'hyperparameters': self.current_hyperparams.copy()
        }
        
        observation = self._get_observation()
        print(f"New observation: {observation}")
        print(f"Reward: {reward}")
        
        return observation, reward, done, truncated, info

    def _get_observation(self):
        return np.array([
            self.current_hyperparams['learning_rate'],
            self.current_hyperparams['layer_sizes'][0] / 2048,  # Normalize
            self.current_hyperparams['dropout_rate']
        ])

    def close(self):
        print("Closing environment...")
        wandb.finish()
        pass
