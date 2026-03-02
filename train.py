import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from dataset import load_data, ETDataset
from model import TimeSeriesTransformer
from utils import rmse, mae, EarlyStopping


def train():
    config = Config()
    
    data, scaler = load_data(config.DATA_PATH, config.TARGET_COL)
    
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.1)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    train_dataset = ETDataset(train_data, config.INPUT_WINDOW, config.OUTPUT_WINDOW)
    val_dataset = ETDataset(val_data, config.INPUT_WINDOW, config.OUTPUT_WINDOW)
    test_dataset = ETDataset(test_data, config.INPUT_WINDOW, config.OUTPUT_WINDOW)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    input_dim = train_dataset.X.shape[-1]
    model = TimeSeriesTransformer(input_dim, config).to(config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    
    early_stopping = EarlyStopping(patience=10)
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            y = y[:, :, -1]  # target only
            
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.DEVICE), y.to(config.DEVICE)
                y = y[:, :, -1]
                
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    evaluate(model, test_loader, config)


def evaluate(model, test_loader, config):
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(config.DEVICE)
            output = model(X)
            preds.append(output.cpu().numpy())
            trues.append(y[:, :, -1].numpy())
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    print("Test RMSE:", rmse(preds, trues))
    print("Test MAE:", mae(preds, trues))
    
    plt.figure()
    plt.plot(trues[0], label="True")
    plt.plot(preds[0], label="Pred")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()
