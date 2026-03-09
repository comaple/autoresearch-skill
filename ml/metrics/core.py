import torch
import torch.nn.functional as F


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds.argmax(dim=-1) == targets).float().mean().item()


def f1_score(preds: torch.Tensor, targets: torch.Tensor, average: str = "macro") -> float:
    preds = preds.argmax(dim=-1)
    if average == "macro":
        classes = torch.unique(targets)
        f1s = []
        for c in classes:
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1s.append(2 * p * r / (p + r + 1e-8))
        return sum(f1s) / len(f1s)
    return 0.0


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(preds, targets)).item()


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return F.l1_loss(preds, targets).item()


def mape(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (torch.abs((targets - preds) / (targets + 1e-8)) * 100).mean().item()
