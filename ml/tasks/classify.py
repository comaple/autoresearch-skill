import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time


def train_classification(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float = 1e-3,
    device: str = "cuda",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    from ml.metrics.core import accuracy, f1_score
    
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                val_preds.append(out)
                val_targets.append(y)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        acc = accuracy(val_preds, val_targets)
        f1 = f1_score(val_preds, val_targets)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pt")
    
    return best_acc
