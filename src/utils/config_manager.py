import os
import yaml
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from .env_detector import EnvironmentDetector

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration loading and optimization based on the detected environment.
    Ensures speed optimization while maintaining stability.
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config
    
    @staticmethod
    def get_optimized_config(config_path: str, override_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration and optimize it for the current environment
        
        Args:
            config_path: Path to the base configuration file
            override_params: Optional parameters to override in the base config
            
        Returns:
            Optimized configuration dictionary
        """
        # Load base configuration
        config = ConfigManager.load_config(config_path)
        
        # Apply any override parameters
        if override_params:
            ConfigManager._deep_update(config, override_params)
        
        # Get system information
        system_info = EnvironmentDetector.get_system_info()
        
        # Log system information
        logger.info(f"Detected system: {system_info['platform']} | "
                   f"Python: {system_info['python_version']} | "
                   f"GPU: {'Available' if system_info['gpu_available'] else 'Not available'} | "
                   f"Environment: {'Google Colab' if system_info['is_colab'] else 'Local system'}")
        
        # Optimize configuration based on environment
        optimized_config = EnvironmentDetector.optimize_config(config, system_info)
        
        # Log key optimized parameters
        if "training" in optimized_config:
            logger.info(f"Optimized batch size: {optimized_config['training'].get('batch_size')}")
            logger.info(f"Using AMP: {optimized_config['training'].get('amp_enabled', False)}")
        
        if "data" in optimized_config:
            logger.info(f"Data loading workers: {optimized_config['data'].get('num_workers')}")
        
        # Create required folders
        ConfigManager._initialize_required_folders(optimized_config)
        
        return optimized_config
    
    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary with values from another dictionary
        
        Args:
            base_dict: Dictionary to be updated
            update_dict: Dictionary with values to update
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_dir: str, filename: Optional[str] = None) -> str:
        """
        Save configuration to a YAML file
        
        Args:
            config: Configuration dictionary to save
            output_dir: Directory to save configuration to
            filename: Optional filename (default: config_<timestamp>.yaml)
            
        Returns:
            Path to saved configuration file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"config_{timestamp}.yaml"
            
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
            
        logger.info(f"Saved configuration to {output_path}")
        return output_path
    
    @staticmethod
    def export_training_history(history: Dict[str, List], output_dir: str) -> str:
        """
        Export training history to JSON file
        
        Args:
            history: Training history dictionary
            output_dir: Directory to save JSON file
            
        Returns:
            Path to saved JSON file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_history_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            if hasattr(values, 'tolist'):
                serializable_history[key] = values.tolist()
            else:
                serializable_history[key] = values
        
        with open(output_path, 'w') as file:
            json.dump(serializable_history, file, indent=2)
            
        logger.info(f"Saved training history to {output_path}")
        return output_path
    
    @staticmethod
    def _initialize_required_folders(config: Dict[str, Any]) -> None:
        """
        Initialize required folders specified in configuration
        
        Args:
            config: Configuration dictionary
        """
        # Ensure output directory exists
        output_dir = config.get('output', {}).get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure checkpoint directory exists
        checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Ensure logs directory exists
        log_dir = config.get('logging', {}).get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Ensure model directory exists
        model_dir = config.get('model', {}).get('save_dir', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Initialized required folders: {output_dir}, {checkpoint_dir}, {log_dir}, {model_dir}")