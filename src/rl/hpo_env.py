import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ..models.cnn_model import FlexibleCNN
from ..training.trainer import ModelTrainer

class HPOEnvironment(gym.Env):
    def __init__(self, trainer, train_loader, val_loader, num_classes):
        super(HPOEnvironment, self).__init__()
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0001, 64, 0.1]),  # lr, layer_size, dropout
            high=np.array([0.1, 2048, 0.9]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3,),  # Current lr, layer_size, dropout
            dtype=np.float32
        )
        
        self.current_hyperparams = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset model with default hyperparameters
        self.current_hyperparams = {
            'learning_rate': 0.001,
            'layer_sizes': [512],
            'dropout_rate': 0.5
        }
        
        # Reinitialize model with default hyperparameters
        model = FlexibleCNN(
            num_classes=self.num_classes,
            hyperparams=self.current_hyperparams
        ).to(self.trainer.device)
        
        self.trainer.model = model
        self.trainer.optimizer = model.get_optimizer()
        
        return self._get_observation(), {}

    def step(self, action):
        # Update hyperparameters
        self.current_hyperparams['learning_rate'] = float(action[0])
        self.current_hyperparams['layer_sizes'] = [int(action[1])]
        self.current_hyperparams['dropout_rate'] = float(action[2])
        
        # Reinitialize model with new hyperparameters
        model = FlexibleCNN(
            num_classes=self.num_classes,
            hyperparams=self.current_hyperparams
        ).to(self.trainer.device)
        
        self.trainer.model = model
        self.trainer.optimizer = model.get_optimizer()
        
        # Train and evaluate model
        try:
            val_accuracy = self.trainer.train(epochs=1)
            reward = val_accuracy
        except Exception as e:
            print(f"Error during training: {str(e)}")
            reward = 0.0
        
        done = False
        truncated = False
        info = {
            'val_accuracy': val_accuracy,
            'hyperparameters': self.current_hyperparams.copy()
        }
        
        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self):
        return np.array([
            self.current_hyperparams['learning_rate'],
            self.current_hyperparams['layer_sizes'][0] / 2048,  # Normalize
            self.current_hyperparams['dropout_rate']
        ])

    def close(self):
        pass
