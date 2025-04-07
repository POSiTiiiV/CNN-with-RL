import sys
import os
import platform
import multiprocessing
import logging
from typing import Dict, Any, List, Optional, Tuple
import time
import subprocess

logger = logging.getLogger(__name__)

class EnvironmentDetector:
    """
    Detects the current execution environment and optimizes configurations accordingly
    to maximize speed while maintaining stability.
    """
    
    @staticmethod
    def is_colab() -> bool:
        """Check if code is running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @staticmethod
    def has_gpu() -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @staticmethod
    def is_t4_gpu() -> bool:
        """Check if the GPU is a Tesla T4 (commonly available in Colab)"""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            
            # Get GPU device name
            device_name = torch.cuda.get_device_name(0).lower()
            return "t4" in device_name
        except:
            return False
    
    @staticmethod
    def get_available_memory() -> Dict[str, float]:
        """Get available system memory in GB with safety margin applied"""
        try:
            import psutil
            import torch
            
            system_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
            # Apply 20% safety margin to system memory
            system_memory = system_memory * 0.8
            
            gpu_memory = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # Apply 30% safety margin to GPU memory to avoid OOM issues
                gpu_memory = gpu_memory * 0.7
            
            return {
                'system': round(system_memory, 2),
                'gpu': round(gpu_memory, 2)
            }
        except ImportError:
            return {'system': 6.0, 'gpu': 0.0}  # Default conservative values with safety margin
    
    @staticmethod
    def get_num_cpus() -> int:
        """Get number of available CPU cores"""
        return multiprocessing.cpu_count()
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Gather system information"""
        is_colab_env = EnvironmentDetector.is_colab()
        gpu_available = EnvironmentDetector.has_gpu()
        is_t4_gpu = EnvironmentDetector.is_t4_gpu() if gpu_available else False
        num_cpus = EnvironmentDetector.get_num_cpus()
        memory = EnvironmentDetector.get_available_memory()
        
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'is_colab': is_colab_env,
            'gpu_available': gpu_available,
            'is_t4_gpu': is_t4_gpu,
            'num_cpus': num_cpus,
            'memory': memory
        }
    
    @staticmethod
    def ensure_checkpoint_dirs(config: Dict[str, Any]) -> None:
        """
        Ensure that checkpoint directories exist
        
        Args:
            config: Configuration dictionary with checkpoint paths
        """
        checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints')
        model_dir = config.get('model', {}).get('save_dir', 'models')
        
        for directory in [checkpoint_dir, model_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
    
    @staticmethod
    def install_missing_packages(required_packages: List[str]) -> None:
        """
        Install missing packages required for the project
        
        Args:
            required_packages: List of required package names
        """
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.split('>=')[0].split('==')[0].strip())
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info("Package installation complete")
    
    @staticmethod
    def optimize_config(config: Dict[str, Any], system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize configuration based on detected environment
        
        Args:
            config: Base configuration dictionary
            system_info: System information dictionary
        
        Returns:
            Optimized configuration dictionary
        """
        optimized_config = config.copy()
        
        # Create training section if it doesn't exist
        if 'training' not in optimized_config:
            optimized_config['training'] = {}
            
        # Create data section if it doesn't exist
        if 'data' not in optimized_config:
            optimized_config['data'] = {}
            
        # Data loading optimization
        optimized_config['data']['num_workers'] = max(1, min(system_info['num_cpus'] - 1, 4))
        optimized_config['data']['pin_memory'] = system_info['gpu_available']
        
        # Batch size optimization
        if system_info['is_colab']:
            # Colab usually has good GPU but limited CPU resources
            if system_info['gpu_available']:
                # Special optimization for Tesla T4 GPU with 15.83GB memory
                if system_info['is_t4_gpu']:
                    logger.info("Tesla T4 GPU detected in Colab - applying T4-specific optimizations")
                    # T4 has substantial memory, so we can use larger batch sizes
                    optimized_config['training']['batch_size'] = 64  # Start with larger batch size for T4
                    optimized_config['training']['amp_enabled'] = True  # Always use mixed precision
                    optimized_config['data']['cache_images'] = False  # Don't cache PIL images
                    optimized_config['data']['cache_tensors'] = True  # Cache tensors directly
                    optimized_config['data']['max_cache_size_gb'] = 6  # T4 has more memory, allow larger cache
                    optimized_config['data']['prefetch_factor'] = 3  # Increase prefetch for T4
                    optimized_config['training']['gradient_accumulation_steps'] = 2  # Fewer accumulation steps needed
                    # Add T4-specific optimizations
                    optimized_config['training']['use_mixed_precision'] = True  # Enable mixed precision training
                    optimized_config['training']['dynamic_batch_sizing'] = True  # Keep dynamic sizing for safety
                else:
                    # For other Colab GPUs: use smaller batches initially and rely on dynamic batch sizing
                    # This prevents early OOM crashes
                    optimized_config['training']['batch_size'] = 32  # Start with moderate batch
                    optimized_config['training']['amp_enabled'] = True  # Always use mixed precision in Colab
                    # Colab-specific optimizations for memory usage
                    optimized_config['data']['cache_images'] = False  # Don't cache PIL images
                    optimized_config['data']['cache_tensors'] = True  # Cache tensors directly
                    optimized_config['data']['max_cache_size_gb'] = 2  # Limit cache size
                    optimized_config['data']['prefetch_factor'] = 2  # Lower prefetch to reduce memory pressure
                    optimized_config['training']['gradient_accumulation_steps'] = 4  # More accumulation steps for smaller GPUs
            else:
                # CPU-only Colab - be very conservative
                optimized_config['training']['batch_size'] = 8
                optimized_config['training']['amp_enabled'] = False
                optimized_config['data']['cache_images'] = False
                optimized_config['data']['cache_tensors'] = False  # Disable caching entirely
                optimized_config['training']['gradient_accumulation_steps'] = 8  # More accumulation steps for CPU
        else:
            # Local system - be adaptive based on available resources
            if system_info['gpu_available']:
                # GPU on local system - optimize based on memory
                if system_info['memory']['gpu'] > 8:
                    # High-end GPU
                    optimized_config['training']['batch_size'] = max(64, optimized_config['training'].get('batch_size', 64))
                    optimized_config['training']['amp_enabled'] = True
                else:
                    # Lower-end GPU
                    optimized_config['training']['batch_size'] = min(32, optimized_config['training'].get('batch_size', 32))
                    optimized_config['training']['amp_enabled'] = True
            else:
                # CPU-only local system
                optimized_config['training']['batch_size'] = min(8, optimized_config['training'].get('batch_size', 8))
                optimized_config['training']['amp_enabled'] = False
        
        # Make sure prefetch factor is reasonable
        if 'prefetch_factor' not in optimized_config['data']:
            if system_info['is_t4_gpu']:
                optimized_config['data']['prefetch_factor'] = 3
            elif system_info['is_colab']:
                optimized_config['data']['prefetch_factor'] = 2
            else:
                optimized_config['data']['prefetch_factor'] = 2
        
        # Stability settings
        optimized_config['training']['dynamic_batch_sizing'] = True  # Enable dynamic batch sizing for OOM prevention
        
        # Set checkpoint frequency based on environment (more frequent on less stable environments)
        optimized_config['training']['checkpoint_freq'] = 3 if system_info['is_colab'] else 10
        
        # Create checkpoint directories
        EnvironmentDetector.ensure_checkpoint_dirs(optimized_config)
        
        # Log optimizations
        logger.info(f"Environment optimizations applied for "
                   f"{'Google Colab' if system_info['is_colab'] else 'Local System'} "
                   f"with {'GPU' if system_info['gpu_available'] else 'CPU'}")
        
        if system_info['is_t4_gpu']:
            logger.info("Tesla T4 GPU optimizations applied - using larger batch sizes and memory cache")
        elif system_info['is_colab']:
            logger.info("Colab-specific optimizations applied to prevent memory issues")
        
        return optimized_config
    
    class ResourceMonitor:
        """
        Monitor system resources during training to prevent OOM errors and optimize performance
        """
        def __init__(self, check_interval: int = 10):
            """
            Initialize the resource monitor
            
            Args:
                check_interval: Interval in seconds between resource checks
            """
            self.check_interval = check_interval
            self.start_time = time.time()
            self.last_check_time = self.start_time
            self.memory_history = []
            self.running = False
            
        def start(self) -> None:
            """Start monitoring resources"""
            self.running = True
            self.start_time = time.time()
            self.last_check_time = self.start_time
            self.memory_history = []
            
        def stop(self) -> None:
            """Stop monitoring resources"""
            self.running = False
            
        def check_resources(self) -> Optional[Dict[str, Any]]:
            """
            Check current resource usage
            
            Returns:
                Dictionary with current resource usage or None if not time to check yet
            """
            current_time = time.time()
            
            # Only check resources at specified intervals
            if current_time - self.last_check_time < self.check_interval:
                return None
                
            self.last_check_time = current_time
            
            try:
                import psutil
                import torch
                
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Get RAM usage
                ram = psutil.virtual_memory()
                ram_used_gb = ram.used / (1024 ** 3)
                ram_total_gb = ram.total / (1024 ** 3)
                ram_percent = ram.percent
                
                result = {
                    'cpu_percent': cpu_percent,
                    'ram_used_gb': round(ram_used_gb, 2),
                    'ram_total_gb': round(ram_total_gb, 2),
                    'ram_percent': ram_percent,
                    'gpu_info': {}
                }
                
                # Get GPU usage if available
                if torch.cuda.is_available():
                    # Current GPU memory usage
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                    gpu_max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    
                    result['gpu_info'] = {
                        'gpu_memory_allocated_gb': round(gpu_memory_allocated, 2),
                        'gpu_memory_reserved_gb': round(gpu_memory_reserved, 2),
                        'gpu_max_memory_gb': round(gpu_max_memory, 2),
                        'gpu_utilization_percent': round((gpu_memory_allocated / gpu_max_memory) * 100, 2)
                    }
                
                self.memory_history.append({
                    'timestamp': current_time - self.start_time,
                    'ram_percent': ram_percent,
                    'gpu_percent': result['gpu_info'].get('gpu_utilization_percent', 0) if 'gpu_info' in result else 0
                })
                
                return result
            except ImportError:
                logger.warning("psutil or torch not available for resource monitoring")
                return None
            except Exception as e:
                logger.warning(f"Error monitoring resources: {str(e)}")
                return None
                
        def get_memory_history(self) -> List[Dict[str, Any]]:
            """
            Get memory usage history
            
            Returns:
                List of dictionaries with memory usage history
            """
            return self.memory_history
            
        def should_reduce_batch_size(self, threshold: float = 90.0) -> bool:
            """
            Check if batch size should be reduced based on resource usage
            
            Args:
                threshold: Memory usage threshold percentage above which batch size should be reduced
                
            Returns:
                True if batch size should be reduced, False otherwise
            """
            resources = self.check_resources()
            if resources is None:
                return False
                
            # Check if any memory usage is above threshold
            ram_above_threshold = resources['ram_percent'] > threshold
            gpu_above_threshold = False
            
            if 'gpu_info' in resources and resources['gpu_info']:
                gpu_above_threshold = resources['gpu_info'].get('gpu_utilization_percent', 0) > threshold
                
            return ram_above_threshold or gpu_above_threshold