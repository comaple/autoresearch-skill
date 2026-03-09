import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_regression(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float = 1e-3,
    device: str = "cuda",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    from ml.metrics.core import rmse, mae
    
    best_rmse = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x).squeeze()
                val_preds.append(out)
                val_targets.append(y)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_rmse = rmse(val_preds, val_targets)
        val_mae = mae(val_preds, val_targets)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss/len(train_loader):.4f} | Val RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), "best_model.pt")
    
    return best_rmse
