import os
import yaml
import logging
from typing import Dict, Any, Optional
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