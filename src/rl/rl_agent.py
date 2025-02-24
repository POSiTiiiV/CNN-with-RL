import numpy as np

class RLAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.intervention_history = []
        self.min_epochs_between_interventions = 3
        self.last_intervention_epoch = -self.min_epochs_between_interventions
    
    def should_intervene(self, epoch, val_metrics, training_history):
        """Decide whether to intervene based on training progress"""
        if epoch < 5:  # Don't intervene during first 5 epochs
            return False
            
        if epoch - self.last_intervention_epoch < self.min_epochs_between_interventions:
            return False
            
        # Check for performance plateau
        if len(training_history) >= 3:
            recent_vals = [h['val_acc'] for h in training_history[-3:]]
            if max(recent_vals) - min(recent_vals) < 0.5:  # Less than 0.5% improvement
                self.last_intervention_epoch = epoch
                return True
                
        return False
    
    def get_intervention_action(self, current_metrics):
        """Generate new hyperparameters when intervening"""
        # Simple exploration strategy
        action = self.action_space.sample()
        
        # Record intervention
        self.intervention_history.append({
            'epoch': current_metrics['epoch'],
            'val_acc': current_metrics['val_acc'],
            'action': action
        })
        
        return action
