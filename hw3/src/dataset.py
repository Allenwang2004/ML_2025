# dataset.py
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FacialExpressionDataset(Dataset):
    def __init__(self,data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(csv_path, valid_ratio=0.2, seed=42):
    import pandas as pd
    import random
    random.seed(seed)
    
    df = pd.read_csv(csv_path)
    all_data = list(zip(df['path'], df['label']))
    random.shuffle(all_data)

    split_idx = int(len(all_data) * (1 - valid_ratio))
    train_data = all_data[:split_idx]
    valid_data = all_data[split_idx:]
    return train_data, valid_data
           