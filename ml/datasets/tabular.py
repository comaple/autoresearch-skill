import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path


class TabularDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str, task_type: str = "regression"):
        self.data = pd.read_csv(csv_path)
        self.target_col = target_col
        self.task_type = task_type
        
        self.features = self.data.drop(columns=[target_col]).values.astype(np.float32)
        self.targets = self.data[target_col].values
        
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.mean) / self.std
        
        if task_type == "classification":
            self.classes = np.unique(self.targets)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.targets = np.array([self.class_to_idx[c] for c in self.targets])
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.long if self.task_type == "classification" else torch.float32)
        return x, y


class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str, seq_len: int = 64, pred_len: int = 16):
        self.data = pd.read_csv(csv_path)
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.feature_cols = [c for c in self.data.columns if c != target_col]
        
        all_data = self.data.values.astype(np.float32)
        self.mean = all_data.mean(axis=0)
        self.std = all_data.std(axis=0) + 1e-8
        self.data_norm = (all_data - self.mean) / self.std
        
        target_idx = self.data.columns.get_loc(target_col)
        self.target_idx = target_idx
        
        self.sequences = []
        self.labels = []
        
        for i in range(len(self.data_norm) - seq_len - pred_len + 1):
            seq = self.data_norm[i:i+seq_len]
            label = self.data_norm[i+seq_len:i+seq_len+pred_len, target_idx]
            self.sequences.append(seq)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.image_dir.glob("**/*.jpg")) + list(self.image_dir.glob("**/*.png")))
        
        self.classes = sorted(list(set(p.parent.name for p in self.image_paths if p.parent != self.image_dir)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[img_path.parent.name] if img_path.parent != self.image_dir else 0
        return image, label
