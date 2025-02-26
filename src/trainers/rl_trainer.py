import os
import json
from datetime import datetime

class RLTrainer:
    def __init__(self, agent, env, cnn_trainer, config):
        """
        Trainer for RL-based hyperparameter optimization
        
        Args:
            agent: RLAgent instance
            env: Hyperparameter optimization environment
            cnn_trainer: CNNTrainer instance to be optimized
            config: Configuration dictionary
        """
        self.agent = agent
        self.env = env
        self.cnn_trainer = cnn_trainer
        self.config = config
        
        # RL training parameters
        self.episodes = config.get('rl_episodes', 100)
        self.steps_per_episode = config.get('steps_per_episode', 10)
        
        # Logging
        self.log_dir = config.get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.history = {
            'episode_rewards': [],
            'best_val_acc': 0,
            'best_hyperparams': {}
        }
    
    def train(self):
        """Train RL agent to find optimal CNN hyperparameters"""
        print("Starting RL-based hyperparameter optimization...")
        
        for episode in range(self.episodes):
            print(f"Episode {episode+1}/{self.episodes}")
            observation, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(self.steps_per_episode):
                # Get action (hyperparameter choices) from agent
                action = self.agent.predict(observation)
                
                # Apply action to environment
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Train the agent
                self.agent.train(1)  # Train for 1 timestep
                
                episode_reward += reward
                observation = next_observation
                
                if done:
                    break
            
            # Log results
            self.history['episode_rewards'].append(episode_reward)
            if info.get('val_acc', 0) > self.history['best_val_acc']:
                self.history['best_val_acc'] = info.get('val_acc', 0)
                self.history['best_hyperparams'] = info.get('hyperparams', {})
            
            # Save logs
            with open(self.log_file, 'w') as f:
                json.dump(self.history, f, indent=4)
                
            print(f"Episode {episode+1} reward: {episode_reward}")
            print(f"Best validation accuracy: {self.history['best_val_acc']}")
        
        return self.history
    
    def save_agent(self, path):
        """Save agent to disk"""
        self.agent.save(path)
