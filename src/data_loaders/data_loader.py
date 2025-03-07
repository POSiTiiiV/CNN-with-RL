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

# Configure logging
logger = logging.getLogger(__name__)

# Configure multiprocessing method for Windows to avoid shared memory issues
if platform.system() == 'Windows':
    # Set appropriate start method for Windows
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn' for Windows compatibility")
    except RuntimeError:
        logger.warning("Multiprocessing start method already set, cannot change now")

# Global image cache that can be shared between datasets
GLOBAL_IMAGE_CACHE = {}

# Add memory monitoring

def is_memory_critical():
    """Check if system memory is critically low"""
    return psutil.virtual_memory().available < 1 * 1024 * 1024 * 1024  # 1GB threshold
# TODO: is it using the GPU or not?
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
            
        # Log cache size
        if use_global_cache:
            logger.info(f"Using global image cache with {len(GLOBAL_IMAGE_CACHE)} images")
        
        self.config = {}  # Initialize config dictionary to avoid AttributeError

    def _preload_images(self):
        """Preload images into cache"""
        logger.info(f"Preloading images for {self.dataset_type} dataset...")
        start_time = time.time()
        total = len(self.data)
        
        # Add memory usage monitoring
        max_cache_size = self.config.get("max_cache_size_gb", 4) * 1024 * 1024 * 1024  # 4GB default
        current_cache_size = 0
        
        for idx in range(total):
            if idx % 100 == 0 and is_memory_critical():
                logger.warning("System memory critically low. Stopping preloading.")
                break
            
            row = self.data.iloc[idx]
            img_path = os.path.join(self.images_dir, row['filename'])
            filename = row['filename']
            
            # Use filename as key for better sharing across datasets
            cache_key = filename
            
            # Before adding to cache, check size
            if current_cache_size > max_cache_size:
                logger.warning(f"Cache size limit reached ({max_cache_size/1e9:.1f}GB). Stopping preloading.")
                break
            
            if cache_key not in self.images_cache:
                try:
                    image = Image.open(img_path).convert('RGB')
                    
                    # If caching tensors directly, apply transform now
                    if self.cache_tensors and self.transform:
                        tensor_img = self.transform(image)
                        self.tensor_cache[cache_key] = tensor_img
                        # No need to store the PIL image if we're caching the tensor
                        if not self.cache_images:
                            continue
                    
                    if self.cache_images:
                        # Store without making an unnecessary copy
                        self.images_cache[cache_key] = image
                    
                    if (idx + 1) % 500 == 0 or idx == total - 1:
                        elapsed = time.time() - start_time
                        logger.info(f"[{self.dataset_type}] Cached {idx+1}/{total} images ({(idx+1)/total*100:.1f}%) - Time: {elapsed:.1f}s...")
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {str(e)}")
            
            # After adding to cache
            if idx % 100 == 0:
                # Estimate current memory usage
                if self.cache_tensors:
                    # Estimate tensor memory
                    # Each float32 tensor of [3, 224, 224] is about 602KB
                    current_cache_size = len(self.tensor_cache) * 602 * 1024
                    logger.debug(f"Estimated cache size: {current_cache_size/1e6:.1f}MB")
        
        elapsed = time.time() - start_time
        logger.info(f"[{self.dataset_type}] Completed caching in {elapsed:.2f}s - Images: {len(self.images_cache)}, Tensors: {len(self.tensor_cache)}")

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
            logger.warning("Encountered shared memory error, using fallback collate method")
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
        for i, _ in enumerate(dataloader):
            logger.info(f"Processed batch {i+1}/{min(3, len(dataloader))} for warmup")
            if i >= 2:  # Process just 3 batches
                break
        elapsed = time.time() - start_time
        logger.info(f"{name} DataLoader warmup completed in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")

def load_dataset(config, data_dir):
    """Load and prepare the fundus image dataset with optimized performance"""
    start_time = time.time()
    logger.info("Loading fundus image dataset with highly optimized performance...")
    
    # Get configuration values
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["data"].get("num_workers", min(os.cpu_count() or 1, 4))
    image_size = config["data"].get("image_size", 224)
    train_split = config["data"].get("train_val_split", 0.8)
    cache_images = config["data"].get("cache_images", True)
    cache_tensors = config["data"].get("cache_tensors", True)  # New option
    use_global_cache = config["data"].get("use_global_cache", True)  # New option
    minimal_transform = config["data"].get("minimal_transform", True)
    persistent_workers = config["data"].get("persistent_workers", True)  # New option
    
    # CSV file path
    csv_path = os.path.join(os.path.dirname(data_dir), "csv_files/balanced_full_df.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Using CSV file: {csv_path}")
    logger.info(f"Using images directory: {data_dir}")
    logger.info(f"Image caching: {'enabled' if cache_images else 'disabled'}")
    logger.info(f"Tensor caching: {'enabled' if cache_tensors else 'disabled'}")
    logger.info(f"Global cache sharing: {'enabled' if use_global_cache else 'disabled'}")
    logger.info(f"Using {'minimal' if minimal_transform else 'full'} transformations")
    logger.info(f"Persistent workers: {'enabled' if persistent_workers else 'disabled'}")
    logger.info(f"Number of workers: {num_workers}")
    
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
    
    logger.info(f"Dataset splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Set up transformations - only apply minimal transformations for preprocessed images
    train_transform = get_transform(train=True, image_size=image_size, minimal=minimal_transform)
    val_transform = get_transform(train=False, image_size=image_size, minimal=minimal_transform)
    
    # Create datasets with appropriate transforms and caching
    dataset_start = time.time()
    logger.info("Creating datasets...")
    
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
        num_workers = min(num_workers, 1)
        logger.info(f"Windows detected: Limiting workers to {num_workers} to avoid shared memory issues")
    
    # Create data loaders with optimized settings
    loader_start = time.time()
    logger.info("Creating data loaders...")
    
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
        warmup_dataloader(train_loader, name="train")
        warmup_dataloader(val_loader, name="val")
        logger.info(f"DataLoader warmup completed in {time.time() - warmup_start:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Total dataset preparation time: {total_time:.2f}s")
    
    return train_loader, val_loader, test_loader
