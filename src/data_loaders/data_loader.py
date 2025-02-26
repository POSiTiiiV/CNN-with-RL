import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import ast
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)

class FundusDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        """
        Args:
            csv_path (str): Path to balanced_full_df.csv
            images_dir (str): Path to merged_images folder
            transform: Optional transform to be applied
        """
        self.images_dir = images_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform or self._default_transform()
        self.class_mapping = {'N':0, 'D':1, 'G':2, 'C':3, 'A':4, 'H':5, 'M':6, 'O':7}

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])
        
        if not os.path.exists(img_path):
            logger.error(f"Image file not found: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise Exception(f"Error loading image {img_path}: {str(e)}")
        
        if self.transform:
            image = self.transform(image)
            
        # Convert string representation of list to actual list
        label_str = row['labels']
        label_list = ast.literal_eval(label_str)
        
        # Create one-hot encoded tensor
        label = torch.zeros(len(self.class_mapping), dtype=torch.float32)
        for l in label_list:
            label[self.class_mapping[l]] = 1.0
            
        return image, label

def get_transform(train=True, image_size=224):
    """Get data transforms based on training or validation phase"""
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

def load_dataset(config, data_dir):
    """Load and prepare the fundus image dataset"""
    logger.info("Loading fundus image dataset")
    
    # Get configuration values
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["data"].get("num_workers", min(os.cpu_count() or 1, 4))
    image_size = config["data"].get("image_size", 224)
    train_split = config["data"].get("train_val_split", 0.8)
    
    # CSV file path
    csv_path = os.path.join(os.path.dirname(data_dir), "csv_files/balanced_full_df.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Using CSV file: {csv_path}")
    logger.info(f"Using images directory: {data_dir}")
    
    # Set up transformations
    train_transform = get_transform(train=True, image_size=image_size)
    val_transform = get_transform(train=False, image_size=image_size)
    
    # Create full dataset first
    full_dataset = FundusDataset(csv_path, data_dir, transform=None)  # We'll apply transforms after splitting
    
    # Get all indices
    indices = list(range(len(full_dataset)))
    
    # Get labels for stratification
    labels = [full_dataset.data.iloc[i]['labels'] for i in indices]
    
    # Create two splits: train and test+val
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_split,
        stratify=labels,
        random_state=42
    )
    
    # Further split test+val into test and val (50/50)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=0.5,
        random_state=42
    )
    
    # Create datasets with appropriate transforms
    train_dataset = Subset(FundusDataset(csv_path, data_dir, transform=train_transform), train_indices)
    val_dataset = Subset(FundusDataset(csv_path, data_dir, transform=val_transform), val_indices)
    test_dataset = Subset(FundusDataset(csv_path, data_dir, transform=val_transform), test_indices)
    
    logger.info(f"Dataset splits: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Update model configuration to match dataset
    config["model"]["num_classes"] = len(full_dataset.class_mapping)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
