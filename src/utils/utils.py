import numpy as np
from typing import Dict, Any

def create_observation(params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Create an observation from the current state.
    
    Args:
        params: Dictionary containing all required parameters
        
    Returns:
        Dict[str, np.ndarray]: The current state observation
    """
    history = params['history']
    current_hyperparams = params['current_hyperparams']
    best_val_acc = params['best_val_acc']
    best_val_loss = params['best_val_loss']
    current_step = params['current_step']
    max_steps = params['max_steps']
    no_improvement_count = params['no_improvement_count']
    patience = params['patience']
    
    observation = {
        'metrics': np.zeros(18, dtype=np.float32),  # Updated to 18 to include extra observations
        'hyperparams': np.zeros(4, dtype=np.float32)
    }
    
    # Current performance metrics
    if history['val_acc']:
        observation['metrics'][0] = history['val_acc'][-1]  # Current val accuracy
        observation['metrics'][1] = min(1.0, max(0.0, 1.0 - history['val_loss'][-1] / 10))  # Normalized val loss
        observation['metrics'][2] = history['train_acc'][-1] if history['train_acc'] else 0.0
        observation['metrics'][3] = min(1.0, max(0.0, 1.0 - history['train_loss'][-1] / 10))
        
    # Best performance so far
    observation['metrics'][4] = best_val_acc
    observation['metrics'][5] = min(1.0, max(0.0, 1.0 - best_val_loss / 10))
    
    # Current hyperparameters (normalized)
    if current_hyperparams:
        # Learning rate (log scale normalization)
        lr = current_hyperparams.get('learning_rate', 0.001)
        observation['hyperparams'][0] = (np.log10(lr) + 5) / 3  # Log scale from 1e-5 to 1e-2
        
        # Weight decay (log scale normalization)
        wd = current_hyperparams.get('weight_decay', 1e-4)
        observation['hyperparams'][1] = (np.log10(wd) + 6) / 4  # Log scale from 1e-6 to 1e-2
        
        # Dropout rate (linear scale)
        observation['hyperparams'][2] = current_hyperparams.get('dropout_rate', 0.5)
        
        # Optimizer type (one-hot like encoding)
        opt_type = current_hyperparams.get('optimizer_type', 'adam')
        if opt_type == 'adam':
            observation['hyperparams'][3] = 0.33
        elif opt_type == 'sgd':
            observation['hyperparams'][3] = 0.66
        else:  # adamw
            observation['hyperparams'][3] = 1.0
    
    # Episode progress
    observation['metrics'][6] = current_step / max_steps
    
    # No improvement counter (normalized)
    observation['metrics'][7] = no_improvement_count / patience
    
    # Relative improvement from last step
    if len(history['val_acc']) > 1:
        last_acc = history['val_acc'][-2]
        current_acc = history['val_acc'][-1]
        rel_improvement = (current_acc - last_acc) / max(0.01, last_acc)
        observation['metrics'][8] = min(1.0, max(0.0, rel_improvement + 0.5))  # Scale to [0,1]

    # Add space for extra observations (trend data)
    observation['metrics'][14] = 0.0  # Placeholder for improvement rate
    observation['metrics'][15] = 0.0  # Placeholder for loss trend
    observation['metrics'][16] = 0.0  # Placeholder for accuracy trend
    observation['metrics'][17] = 0.0  # Placeholder for stagnation flag
    
    return observation

def enhance_observation_with_trends(observation: Dict[str, np.ndarray], trend_data: dict) -> Dict[str, np.ndarray]:
    """
    Add trend information to the observation for the RL agent.
    
    Args:
        observation: Base observation dictionary
        trend_data: Performance trend metrics
        
    Returns:
        Enhanced observation dictionary
    """
    # Ensure we have space for trend data in the metrics part of the observation
    if len(observation['metrics']) < 18:  # Ensure we have space for trend data
        extended_metrics = np.zeros(18, dtype=np.float32)
        extended_metrics[:len(observation['metrics'])] = observation['metrics']
        observation['metrics'] = extended_metrics
        
    # Add trend data to observation (normalize to roughly [-1, 1] range)
    observation['metrics'][14] = np.clip(trend_data['improvement_rate'] * 10, -1, 1)  # Overall improvement rate
    observation['metrics'][15] = np.clip(trend_data['val_loss_trend'] * 20, -1, 1)  # Loss trend direction
    observation['metrics'][16] = np.clip(trend_data['val_acc_trend'] * 20, -1, 1)   # Accuracy trend direction
    observation['metrics'][17] = 1.0 if trend_data['is_stagnating'] else 0.0        # Stagnation flag
    
    return observation

def calculate_performance_trends(params: Dict[str, Any]) -> dict:
    """
    Calculate performance trends using rolling windows of metrics.
    
    Args:
        params: Dictionary containing all required parameters
        
    Returns:
        dict: Trend data including stagnation status and improvement rates
    """
    history = params['history']
    metric_window_size = params['metric_window_size']
    improvement_threshold = params['improvement_threshold']
    loss_stagnation_threshold = params['loss_stagnation_threshold']
    
    # Default values if not enough history
    trend_data = {
        'is_stagnating': False,
        'improvement_rate': 1.0,  # High default rate to prevent early intervention
        'val_loss_trend': 0.0,
        'val_acc_trend': 0.0,
        'loss_volatility': 0.0,
        'acc_volatility': 0.0
    }
    
    # Need enough history to calculate trends
    if len(history['val_loss']) < metric_window_size:
        return trend_data
        
    # Get recent metrics
    recent_losses = history['val_loss'][-metric_window_size:]
    recent_accs = history['val_acc'][-metric_window_size:]
    
    # Calculate moving average
    loss_ma = sum(recent_losses) / len(recent_losses)
    acc_ma = sum(recent_accs) / len(recent_accs)
    
    # Calculate slopes (trend direction)
    # Positive loss_slope = worsening, Negative loss_slope = improving
    # Positive acc_slope = improving, Negative acc_slope = worsening
    loss_slope = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
    acc_slope = (recent_accs[-1] - recent_accs[0]) / len(recent_accs)
    
    # Calculate volatility (standard deviation)
    loss_volatility = np.std(recent_losses)
    acc_volatility = np.std(recent_accs)
    
    # Average improvement rate (combination of loss and accuracy trends)
    # Normalize both to similar scales (want both to be positive when improving)
    normalized_loss_improvement = -loss_slope / (loss_ma + 1e-6)  # Negative because decreasing loss is good
    normalized_acc_improvement = acc_slope / (acc_ma + 1e-6)
    improvement_rate = (normalized_loss_improvement + normalized_acc_improvement) / 2
    
    # Define stagnation: low improvement rate and low volatility
    is_stagnating = (abs(improvement_rate) < improvement_threshold and 
                      loss_volatility < loss_stagnation_threshold)
    
    return {
        'is_stagnating': is_stagnating,
        'improvement_rate': improvement_rate,
        'val_loss_trend': loss_slope,
        'val_acc_trend': acc_slope,
        'loss_volatility': loss_volatility,
        'acc_volatility': acc_volatility,
        'loss_ma': loss_ma,
        'acc_ma': acc_ma
    }
