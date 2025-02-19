import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import ast

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
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise Exception(f"Error loading image {img_path}: {str(e)}")
        
        if self.transform:
            image = self.transform(image)
            
        # Convert string representation of list to actual list
        label_str = row['labels']
        label_list = ast.literal_eval(label_str)
        
        # Create one-hot encoded tensor
        label = torch.zeros(8, dtype=torch.float32)  # 8 classes as per CSV
        for l in label_list:
            label[['N','D','G','C','A','H','M','O'].index(l)] = 1.0
            
        return image, label

def get_fundus_data_loaders(csv_path, images_dir, batch_size=32, train_split=0.8):
    from sklearn.model_selection import train_test_split
    
    # Create full dataset first to get proper stratification
    dataset = FundusDataset(csv_path, images_dir)
    
    # Get all indices
    indices = list(range(len(dataset)))
    
    # Get labels for stratification
    labels = [dataset.data.iloc[i]['labels'] for i in indices]
    
    # Split indices while maintaining class distribution
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_split,
        stratify=labels,
        random_state=42
    )
    
    # Create train and validation datasets using indices
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
