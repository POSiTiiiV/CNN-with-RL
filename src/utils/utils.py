import numpy as np
import torch
import logging
from typing import Dict, Any

# Set up logger
logger = logging.getLogger("cnn_rl.utils")

def create_observation(params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Create an observation from the current state with integrated trend data.
    
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
    
    # Setup observation arrays (metrics and hyperparams)
    observation = {
        'metrics': np.zeros(18, dtype=np.float32),  # Space for all metrics including trends
        'hyperparams': np.zeros(4, dtype=np.float32)  # Space for normalized hyperparameters
    }
    
    # Current performance metrics - safely access history with null checks
    if history['val_acc'] and len(history['val_acc']) > 0:
        observation['metrics'][0] = history['val_acc'][-1]  # Current val accuracy
    
    if history['val_loss'] and len(history['val_loss']) > 0:
        observation['metrics'][1] = min(1.0, max(0.0, 1.0 - history['val_loss'][-1] / 10))  # Normalized val loss
    
    if history['train_acc'] and len(history['train_acc']) > 0:
        observation['metrics'][2] = history['train_acc'][-1]  # Train accuracy
    
    if history['train_loss'] and len(history['train_loss']) > 0:
        observation['metrics'][3] = min(1.0, max(0.0, 1.0 - history['train_loss'][-1] / 10))  # Normalized train loss
        
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
    
    # Relative improvement from last step - safely check list length
    if history['val_acc'] and len(history['val_acc']) > 1:
        last_acc = history['val_acc'][-2]
        current_acc = history['val_acc'][-1]
        rel_improvement = (current_acc - last_acc) / max(0.01, last_acc)
        observation['metrics'][8] = min(1.0, max(0.0, rel_improvement + 0.5))  # Scale to [0,1]

    # Reserve space for trend data (will be filled by enhance_observation_with_trends)
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
    metric_window_size = params.get('metric_window_size', 5)
    improvement_threshold = params.get('improvement_threshold', 0.002)
    loss_stagnation_threshold = params.get('loss_stagnation_threshold', 0.003)
    
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

def is_significant_hyperparameter_change(new_hyperparams, current_hyperparams):
    """
    Determine if the hyperparameter changes are significant.
    
    Args:
        new_hyperparams: New hyperparameters
        current_hyperparams: Currently used hyperparameters
        
    Returns:
        tuple: (is_significant, is_major_change, reason)
            - is_significant: True if any change is significant
            - is_major_change: True if structural changes (requiring model rebuild)
            - reason: Description of the major change or None
    """
    if not current_hyperparams:
        return True, True, "Initial hyperparameters"

    # Check for major structural changes first
    major_changes = None
    
    # Check for FC layer configuration changes
    if new_hyperparams.get('fc_layers') != current_hyperparams.get('fc_layers'):
        major_changes = f"FC layer configuration changed from {current_hyperparams.get('fc_layers')} to {new_hyperparams.get('fc_layers')}"
        return True, True, major_changes
        
    # Check for optimizer type changes
    if new_hyperparams.get('optimizer_type') != current_hyperparams.get('optimizer_type'):
        major_changes = f"Optimizer type changed from {current_hyperparams.get('optimizer_type')} to {new_hyperparams.get('optimizer_type')}"
        return True, True, major_changes
    
    # For non-major changes, check if they're significant enough
    has_significant_change = False
    for key, new_value in new_hyperparams.items():
        if key in current_hyperparams:
            current_value = current_hyperparams[key]
            
            # For numeric values, check percentage difference
            if isinstance(new_value, (int, float)) and isinstance(current_value, (int, float)):
                if current_value != 0:
                    # Consider significant if change is more than 5%
                    pct_diff = abs(new_value - current_value) / abs(current_value)
                    if pct_diff > 0.05:  # 5% threshold
                        has_significant_change = True
                        break
                elif new_value != 0:  # Current is 0 but new is not
                    has_significant_change = True
                    break
            # For non-numeric values, direct comparison
            elif new_value != current_value:
                has_significant_change = True
                break
        else:
            # New parameter that wasn't in current set
            has_significant_change = True
            break
                
    return has_significant_change, False, None

def get_hyperparams_hash(hyperparams):
    """
    Create a hash string for a hyperparameter configuration.
    
    Args:
        hyperparams: Hyperparameter dictionary
        
    Returns:
        str: Hash string representing the configuration
    """
    # Round numerical values to avoid minor differences
    lr = round(hyperparams.get('learning_rate', 0), 6)
    wd = round(hyperparams.get('weight_decay', 0), 8)
    dr = round(hyperparams.get('dropout_rate', 0), 2)
    fc = '-'.join([str(x) for x in hyperparams.get('fc_layers', [])])
    opt = hyperparams.get('optimizer_type', '')
    
    return f"lr{lr}_wd{wd}_dr{dr}_fc{fc}_opt{opt}"

def get_optimizer(model, optimizer_type='adam', learning_rate=0.001, weight_decay=0.0001):
    """
    Create optimizer based on hyperparameters.

    Args:
        model: The model whose parameters will be optimized
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay (L2 regularization) for the optimizer

    Returns:
        torch.optim.Optimizer: The optimizer
    """
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.warning(f"[yellow]Warning: Unknown optimizer type: {optimizer_type}, using Adam[/yellow]")
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
