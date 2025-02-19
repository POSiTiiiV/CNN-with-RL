import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

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
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = row['label']  # Adjust this based on your CSV column name
        return image, label

def get_fundus_data_loaders(csv_path, images_dir, batch_size=32, train_split=0.8):
    dataset = FundusDataset(csv_path, images_dir)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
