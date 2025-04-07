import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import ast
import logging
from sklearn.model_selection import train_test_split
import numpy as np
import time
import psutil
import platform
from tqdm import tqdm

# Replace Loguru and Rich logging with Python logging and tqdm
logger = logging.getLogger(__name__)

def config_multiprocessing():
    # Configure multiprocessing method for Windows to avoid shared memory issues
    if platform.system() == 'Windows':
        # Set appropriate start method for Windows
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            logger.warning("Multiprocessing start method already set, cannot change now")

# Global image cache that can be shared between datasets
GLOBAL_IMAGE_CACHE = {}

# Add memory monitoring

def is_memory_critical(colab_threshold=2 * 1024 * 1024 * 1024, local_threshold=1 * 1024 * 1024 * 1024):
    """Check if system memory is critically low with different thresholds for Colab and local environments"""
    try:
        # Check if we're in Colab
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    # Use a more aggressive threshold for Colab (2GB) compared to local (1GB)
    threshold = colab_threshold if is_colab else local_threshold
    return psutil.virtual_memory().available < threshold

class OptimizedFundusDataset(Dataset):
    def __init__(self, df, images_dir, transform=None, cache_images=True, cache_tensors=True, 
                 use_global_cache=True, dataset_type=""):
        """
        Args:
            df (DataFrame): Pandas dataframe with image info
            images_dir (str): Path to merged_images folder
            transform: Optional transform to be applied
            cache_images (bool): Whether to cache images in memory
            cache_tensors (bool): Whether to cache transformed tensors
            use_global_cache (bool): Whether to use the global cache
            dataset_type (str): Type of dataset (train/val/test) for logging
        """
        self.images_dir = images_dir
        self.data = df
        self.transform = transform
        self.class_mapping = {'N':0, 'D':1, 'G':2, 'C':3, 'A':4, 'H':5, 'M':6, 'O':7}
        self.cache_images = cache_images
        self.cache_tensors = cache_tensors
        self.use_global_cache = use_global_cache
        self.dataset_type = dataset_type
        self.local_cache = {} if not use_global_cache else None
        
        # Cache for transformed tensors
        self.tensor_cache = {}
        
        # Reference the global cache if using global cache
        self.images_cache = GLOBAL_IMAGE_CACHE if use_global_cache else self.local_cache
        
        # Preload images if caching is enabled
        if self.cache_images and (len(self.images_cache) == 0 or not use_global_cache):
            self._preload_images()
        
        self.config = {}  # Initialize config dictionary to avoid AttributeError

        config_multiprocessing()  # Configure multiprocessing for Windows

    def _preload_images(self):
        """Preload images into cache"""
        logger.info(f"Preloading images for {self.dataset_type} dataset...")
        start_time = time.time()
        total = len(self.data)
        
        # Detect if we're in Colab for environment-specific settings
        try:
            import google.colab
            is_colab = True
            logger.info("Colab environment detected - using specialized memory management")
        except ImportError:
            is_colab = False
        
        # Adjust max cache size based on environment
        default_cache_size = 2 if is_colab else 4  # 2GB for Colab, 4GB for local
        max_cache_size = self.config.get("max_cache_size_gb", default_cache_size) * 1024 * 1024 * 1024
        current_cache_size = 0
        
        # Colab environments need more frequent memory checks
        check_interval = 20 if is_colab else 100
        
        with tqdm(total=total, desc=f"Caching {self.dataset_type} images", unit="image") as progress:
            for idx in range(total):
                if idx % check_interval == 0 and is_memory_critical():
                    logger.warning("⚠️ System memory getting low. Stopping preloading.")
                    break
                
                row = self.data.iloc[idx]
                img_path = os.path.join(self.images_dir, row['filename'])
                filename = row['filename']
                
                # Use filename as key for better sharing across datasets
                cache_key = filename
                
                # Check if we're approaching cache size limit
                if current_cache_size > max_cache_size:
                    logger.warning(f"⚠️ Cache size limit reached ({max_cache_size/1e9:.1f}GB). Stopping preloading.")
                    break
                
                if cache_key not in self.images_cache:
                    try:
                        image = Image.open(img_path).convert('RGB')
                        
                        # In Colab, prefer tensor caching over PIL image caching to optimize GPU usage
                        if is_colab:
                            self.cache_images = False
                            self.cache_tensors = True
                        
                        # If caching tensors directly, apply transform now
                        if self.cache_tensors and self.transform:
                            tensor_img = self.transform(image)
                            self.tensor_cache[cache_key] = tensor_img
                            # No need to store the PIL image if we're caching the tensor
                            if not self.cache_images:
                                # Force image cleanup
                                image.close()
                                del image
                                continue
                        
                        if self.cache_images:
                            # Store without making an unnecessary copy
                            self.images_cache[cache_key] = image
                        
                        progress.update(1)
                    except Exception as e:
                        logger.error(f"Error loading image {img_path}: {str(e)}")
                else:
                    progress.update(1)
                
                # More accurate memory usage estimation
                if idx % check_interval == 0:
                    # Calculate current usage more precisely using psutil
                    ram_used = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
                    logger.debug(f"Current RAM usage: {ram_used:.2f}GB")
                    
                    # Force Python garbage collection occasionally
                    if is_colab and idx % (check_interval * 5) == 0:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        
        # Final memory cleanup
        if is_colab:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        logger.info(f"Completed caching in {elapsed:.2f}s - Images: {len(self.images_cache)}, Tensors: {len(self.tensor_cache)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        cache_key = filename
        
        # First check if we have the tensor already cached
        if self.cache_tensors and cache_key in self.tensor_cache:
            # Use the precomputed tensor directly - no copy needed for tensors
            image = self.tensor_cache[cache_key]
        else:
            # Get image from cache if available, otherwise load from disk
            if self.cache_images and cache_key in self.images_cache:
                # No need for .copy() - PyTorch transforms work on a new tensor anyway
                image = self.images_cache[cache_key]
            else:
                img_path = os.path.join(self.images_dir, row['filename'])
                if not os.path.exists(img_path):
                    logger.error(f"Image file not found: {img_path}")
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                    
                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {str(e)}")
                    raise Exception(f"Error loading image {img_path}: {str(e)}")
            
            # Apply transform if it wasn't already pre-applied
            if self.transform:
                image = self.transform(image)
            
            # Cache the transformed tensor for future use
            if self.cache_tensors:
                self.tensor_cache[cache_key] = image
            
        # Convert string representation of list to actual list
        label_str = row['labels']
        label_list = ast.literal_eval(label_str)
        
        # Create one-hot encoded tensor
        label = torch.zeros(len(self.class_mapping), dtype=torch.float32)
        for l in label_list:
            label[self.class_mapping[l]] = 1.0
            
        return image, label

# Custom collate function to avoid shared memory issues on Windows
def safe_collate(batch):
    """
    Custom collate function that handles the shared memory issues on Windows
    by avoiding the use of _new_shared method for large tensors.
    """
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        if "Couldn't open shared file mapping" in str(e) and platform.system() == 'Windows':
            logger.warning("Warning: Encountered shared memory error, using fallback collate method")
            # Fallback method - manually stack tensors without shared memory
            images = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            return images, labels
        else:
            # Re-raise if it's not the specific error we're handling
            raise

def get_transform(train=True, image_size=224, minimal=True):
    """
    Get minimal transformations for preprocessed images
    
    Args:
        train (bool): Whether transformations are for training
        image_size (int): Target image size (might be skipped if minimal=True)
        minimal (bool): If True, use only necessary transformations
    """
    if minimal:
        # Minimal transformations for preprocessed images
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Original transformations if needed
        if train:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

def warmup_dataloader(dataloader, name=""):
    """Warmup dataloader workers by fetching a batch"""
    logger.info(f"Warming up {name} dataloader...")
    start_time = time.time()
    try:
        # Remove tqdm progress bar and process silently
        batch_count = 0
        for _, _ in dataloader:
            batch_count += 1
            if batch_count >= 3:  # Process just 3 batches
                break
        elapsed = time.time() - start_time
        logger.info(f"{name} DataLoader warmup completed in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")

def load_dataset(config, data_dir):
    """Load and prepare the fundus image dataset with optimized performance"""
    start_time = time.time()
    print('\n')
    logger.info("Loading fundus image dataset with highly optimized performance...")
    print('\n')
    
    # Detect if we're using a Tesla T4 GPU in Colab
    try:
        import google.colab
        import torch
        is_colab = True
        is_t4_gpu = False
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()
            is_t4_gpu = "t4" in device_name
            if is_t4_gpu:
                logger.info("Tesla T4 GPU detected in Colab - applying T4-specific data loading optimizations")
    except ImportError:
        is_colab = False
        is_t4_gpu = False
    
    # Get configuration values
    batch_size = config["training"].get("batch_size", 32)
    
    # T4-specific optimizations for Colab
    if is_t4_gpu:
        # T4 has 16GB of memory, so we can use more aggressive settings
        num_workers = min(4, config["data"].get("num_workers", 4))  # T4 can handle more workers
        cache_images = False  # Don't cache PIL images
        cache_tensors = True  # Cache tensors directly
        use_global_cache = True  # Enable global cache sharing
        max_cache_size_gb = config["data"].get("max_cache_size_gb", 6)
        prefetch_factor = config["data"].get("prefetch_factor", 3)
        persistent_workers = True
        logger.info(f"T4-optimized settings - batch_size: {batch_size}, workers: {num_workers}, cache: {max_cache_size_gb}GB")
    else:
        # Regular optimization
        num_workers = config["data"].get("num_workers", min(os.cpu_count() or 1, 4))
        cache_images = config["data"].get("cache_images", True)
        cache_tensors = config["data"].get("cache_tensors", True)
        use_global_cache = config["data"].get("use_global_cache", True)
        max_cache_size_gb = config["data"].get("max_cache_size_gb", 4)
        prefetch_factor = config["data"].get("prefetch_factor", 2)
        persistent_workers = config["data"].get("persistent_workers", True)
    
    image_size = config["data"].get("image_size", 224)
    train_split = config["data"].get("train_val_split", 0.8)
    minimal_transform = config["data"].get("minimal_transform", True)
    
    # CSV file path
    csv_path = os.path.join(os.path.dirname(data_dir), "csv_files/balanced_full_df.csv")
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Log dataset settings
    logger.info(f"Dataset Settings:\n"
                f"CSV file: {csv_path}\n"
                f"Images directory: {data_dir}\n"
                f"Batch size: {batch_size}\n"
                f"Number of workers: {num_workers}\n"
                f"Image size: {image_size}\n"
                f"Train/Val split: {train_split:.2f}\n"
                f"Image caching: {'enabled' if cache_images else 'disabled'}\n"
                f"Tensor caching: {'enabled' if cache_tensors else 'disabled'}\n"
                f"Global cache sharing: {'enabled' if use_global_cache else 'disabled'}\n"
                f"Max cache size: {max_cache_size_gb}GB\n"
                f"Transformations: {'minimal' if minimal_transform else 'full'}\n"
                f"Persistent workers: {'enabled' if persistent_workers else 'disabled'}")
    
    # Load the CSV file once
    df = pd.read_csv(csv_path)
    logger.info(f"CSV loaded: {len(df)} entries in {time.time() - start_time:.2f}s")
    
    # Extract a simple label for stratification
    def get_primary_label(label_str):
        try:
            labels = ast.literal_eval(label_str)
            return labels[0] if labels else "unknown"
        except Exception:
            return "unknown"
    
    df['primary_label'] = df['labels'].apply(get_primary_label)
    
    # Split the data into train, val, and test
    split_start = time.time()
    logger.info("Splitting data into train, validation and test sets...")
    train_df, temp_df = train_test_split(
        df,
        train_size=train_split,
        stratify=df['primary_label'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        stratify=temp_df['primary_label'],
        random_state=42
    )
    logger.info(f"Data split completed in {time.time() - split_start:.2f}s")
    
    # Remove temporary column
    train_df = train_df.drop('primary_label', axis=1)
    val_df = val_df.drop('primary_label', axis=1)
    test_df = test_df.drop('primary_label', axis=1)
    
    print('\n')
    logger.info(f"Dataset splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    print('\n')
    
    # Set up transformations - only apply minimal transformations for preprocessed images
    train_transform = get_transform(train=True, image_size=image_size, minimal=minimal_transform)
    val_transform = get_transform(train=False, image_size=image_size, minimal=minimal_transform)
    
    # Create datasets with appropriate transforms and caching
    dataset_start = time.time()
    print('\n')
    logger.info("Creating datasets...")
    print('\n')
    
    train_dataset = OptimizedFundusDataset(
        train_df, data_dir, transform=train_transform, 
        cache_images=cache_images, cache_tensors=cache_tensors,
        use_global_cache=use_global_cache, dataset_type="train"
    )
    
    val_dataset = OptimizedFundusDataset(
        val_df, data_dir, transform=val_transform, 
        cache_images=cache_images, cache_tensors=cache_tensors,
        use_global_cache=use_global_cache, dataset_type="val"
    )
    
    test_dataset = OptimizedFundusDataset(
        test_df, data_dir, transform=val_transform, 
        cache_images=cache_images, cache_tensors=cache_tensors,
        use_global_cache=use_global_cache, dataset_type="test"
    )
    
    logger.info(f"Datasets created in {time.time() - dataset_start:.2f}s")
    
    # Update model configuration to match dataset
    config["model"]["num_classes"] = len(train_dataset.class_mapping)
    
    # Adjust settings for Windows to avoid shared memory issues
    if platform.system() == 'Windows':
        # Limit number of workers on Windows to avoid shared memory issues
        num_workers = min(num_workers, 2)  # Adjusted to 2 for better performance
        logger.info(f"Windows detected: Limiting workers to {num_workers} to avoid shared memory issues")
    
    # Create data loaders with optimized settings
    loader_start = time.time()
    print('\n')
    logger.info("Creating data loaders...")
    print('\n')
    
    # Use the safe collate function to handle shared memory issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=safe_collate if platform.system() == 'Windows' else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=safe_collate if platform.system() == 'Windows' else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=safe_collate if platform.system() == 'Windows' else None
    )
    
    logger.info(f"Data loaders created in {time.time() - loader_start:.2f}s")
    
    # Warmup the dataloaders if configured
    if config["data"].get("warmup_loaders", True) and num_workers > 0:
        warmup_start = time.time()
        print('\n')
        logger.info("Warming up train dataloader...")
        warmup_dataloader(train_loader, name="train")
        logger.info("Warming up val dataloader...")
        warmup_dataloader(val_loader, name="val")
        logger.info("Warming up test dataloader...")
        warmup_dataloader(test_loader, name="test")
        logger.info("DataLoader warmup completed")
        print('\n')
        logger.info(f"DataLoader warmup completed in {time.time() - warmup_start:.2f}s")
    
    total_time = time.time() - start_time
    logger.info("Total dataset preparation time: {total_time:.2f}s")
    print('\n')
    
    return train_loader, val_loader, test_loader
