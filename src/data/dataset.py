import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class EyeDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)
        self.file_list = self._get_file_list()
        self.transform = transform or self._default_transform()

    def _get_file_list(self):
        files = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            for image_name in os.listdir(class_path):
                files.append((os.path.join(class_path, image_name), self.classes.index(class_name)))
        return files

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path, label = self.file_list[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir, batch_size=32, train_split=0.8):
    dataset = EyeDiseaseDataset(data_dir)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
