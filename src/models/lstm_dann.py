from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .grl import GradientReversal


@dataclass(slots=True)
class LSTMDANNConfig:
    input_size: int
    hidden_size: int = 64
    feature_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    domain_classes: int = 2


class LSTMDANN(nn.Module):
    def __init__(self, config: LSTMDANNConfig) -> None:
        super().__init__()
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.feature_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(config.feature_size, config.feature_size // 2),
            nn.ReLU(),
            nn.Linear(config.feature_size // 2, 1),
        )
        self.grl = GradientReversal()
        self.domain_classifier = nn.Sequential(
            nn.Linear(config.feature_size, config.feature_size // 2),
            nn.ReLU(),
            nn.Linear(config.feature_size // 2, config.domain_classes),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(inputs)
        return self.projection(outputs[:, -1, :])

    def forward(self, inputs: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(inputs)
        rul_prediction = self.regressor(features).squeeze(-1)
        domain_logits = self.domain_classifier(self.grl(features, alpha=alpha))
        return rul_prediction, domain_logits, features
