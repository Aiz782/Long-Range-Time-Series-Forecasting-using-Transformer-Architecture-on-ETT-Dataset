import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ETDataset(Dataset):
    def __init__(self, data, input_window, output_window):
        self.X, self.y = self.create_sequences(
            data, input_window, output_window
        )

    def create_sequences(self, data, input_window, output_window):
        X, y = [], []
        for i in range(len(data) - input_window - output_window):
            X.append(data[i:i+input_window])
            y.append(data[i+input_window:i+input_window+output_window])
        return torch.tensor(X, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(path, target_col):
    df = pd.read_csv(path)
    df = df.drop(columns=["date"], errors="ignore")
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    
    return scaled, scaler
