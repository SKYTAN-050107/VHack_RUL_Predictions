from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluate import rmse


@dataclass(slots=True)
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)


def train_lstm_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 10,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> TrainingHistory:
    """Train a supervised LSTM regressor and optionally track validation RMSE."""
    model.to(device)
    loss_fn = criterion or nn.MSELoss()
    optim = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
    history = TrainingHistory()

    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device).float()

            optim.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())

        history.train_loss.append(float(sum(epoch_losses) / max(len(epoch_losses), 1)))
        if val_loader is not None:
            history.val_rmse.append(evaluate_model(model, val_loader, device=device)["rmse"])

    return history


def train_lstm_dann(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 10,
    optimizer: torch.optim.Optimizer | None = None,
    regression_criterion: nn.Module | None = None,
    domain_criterion: nn.Module | None = None,
    lambda_domain: float = 0.5,
    device: str | torch.device = "cpu",
) -> TrainingHistory:
    """Train an LSTM-DANN using labeled source data and unlabeled target data."""
    model.to(device)
    reg_loss_fn = regression_criterion or nn.MSELoss()
    domain_loss_fn = domain_criterion or nn.CrossEntropyLoss()
    optim = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
    history = TrainingHistory()

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        alpha = float(epoch + 1) / float(max(epochs, 1))

        for (source_x, source_y), (target_x, *_) in zip(source_loader, cycle(target_loader)):
            source_x = source_x.to(device)
            source_y = source_y.to(device).float()
            target_x = target_x.to(device)

            optim.zero_grad()
            source_rul, source_domain_logits, _ = model(source_x, alpha=alpha)
            _, target_domain_logits, _ = model(target_x, alpha=alpha)

            source_domain_labels = torch.zeros(source_x.size(0), dtype=torch.long, device=device)
            target_domain_labels = torch.ones(target_x.size(0), dtype=torch.long, device=device)

            rul_loss = reg_loss_fn(source_rul, source_y)
            domain_loss = domain_loss_fn(source_domain_logits, source_domain_labels)
            domain_loss = domain_loss + domain_loss_fn(target_domain_logits, target_domain_labels)
            total_loss = rul_loss + lambda_domain * domain_loss
            total_loss.backward()
            optim.step()
            epoch_losses.append(total_loss.item())

        history.train_loss.append(float(sum(epoch_losses) / max(len(epoch_losses), 1)))
        if val_loader is not None:
            history.val_rmse.append(evaluate_model(model, val_loader, device=device)["rmse"])

    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Evaluate either a plain regressor or a DANN model on a labeled loader."""
    model.to(device)
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for features, batch_targets in data_loader:
            features = features.to(device)
            batch_targets = batch_targets.to(device).float()
            outputs = model(features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions.append(outputs.detach().cpu())
            targets.append(batch_targets.detach().cpu())

    y_pred = torch.cat(predictions).numpy()
    y_true = torch.cat(targets).numpy()
    return {"rmse": rmse(y_true, y_pred)}


def predict(model: nn.Module, data_loader: DataLoader, device: str | torch.device = "cpu") -> torch.Tensor:
    model.to(device)
    model.eval()
    batches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            outputs = model(features.to(device))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            batches.append(outputs.detach().cpu())

    return torch.cat(batches)
