import numpy as np
from collections import deque

class HPOMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.performance_threshold = 0.0  # Dynamic threshold for good performance

    def add(self, hyperparams, reward, val_accuracy):
        self.memory.append({
            'hyperparams': hyperparams.copy(),
            'reward': reward,
            'val_accuracy': val_accuracy
        })
        # Update performance threshold (moving average of top 25% results)
        if len(self.memory) > 4:
            accuracies = [m['val_accuracy'] for m in self.memory]
            self.performance_threshold = np.percentile(accuracies, 75)

    def get_best_hyperparams(self):
        if not self.memory:
            return None
        best_entry = max(self.memory, key=lambda x: x['reward'])
        return best_entry['hyperparams']

    def get_similarity_score(self, hyperparams):
        if not self.memory:
            return 0.0
        
        # Calculate similarity to previous configurations
        similarities = []
        for entry in self.memory:
            past_params = entry['hyperparams']
            similarity = sum([
                abs(hyperparams['learning_rate'] - past_params['learning_rate']) / past_params['learning_rate'],
                abs(hyperparams['layer_sizes'][0] - past_params['layer_sizes'][0]) / past_params['layer_sizes'][0],
                abs(hyperparams['dropout_rate'] - past_params['dropout_rate']),
                abs(hyperparams['weight_decay'] - past_params['weight_decay']) / past_params['weight_decay']
            ]) / 4.0
            similarities.append((similarity, entry['reward']))
        
        # Weight similarity by past performance
        weighted_similarity = sum(s[0] * s[1] for s in similarities) / sum(s[1] for s in similarities)
        return weighted_similarity
