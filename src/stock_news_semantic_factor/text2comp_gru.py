from __future__ import annotations

import torch
from torch import nn


class Text2Comp(nn.Module):
    """Map LLM text embeddings into compact computation variables."""

    def __init__(self, embedding_dim: int, comp_dim: int = 64, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, comp_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(text_embedding)


class FiLMFusion(nn.Module):
    """Use text computation variables to modulate price hidden states."""

    def __init__(self, comp_dim: int, hidden_dim: int, film_hidden_dims: tuple[int, ...] = (256, 128)):
        super().__init__()
        layers = []
        in_dim = comp_dim
        for layer_dim in film_hidden_dims:
            layers.extend([nn.Linear(in_dim, layer_dim), nn.GELU(), nn.Dropout(0.1)])
            in_dim = layer_dim
        layers.append(nn.Linear(in_dim, hidden_dim * 2))
        self.to_film = nn.Sequential(*layers)

    def forward(self, price_hidden: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_film(comp)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return price_hidden * (1.0 + torch.tanh(gamma)) + beta


class Text2CompGRUClassifier(nn.Module):
    """Stage-1 direction prediction model: OHLCV expert + text computation module."""

    def __init__(
        self,
        price_dim: int = 5,
        embedding_dim: int = 1024,
        gru_hidden_dim: int = 128,
        comp_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.price_encoder = nn.GRU(
            input_size=price_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.text2comp = Text2Comp(embedding_dim=embedding_dim, comp_dim=comp_dim)
        self.fusion = FiLMFusion(comp_dim=comp_dim, hidden_dim=gru_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim + comp_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )

    def forward(self, price_window: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        _, hidden = self.price_encoder(price_window)
        price_hidden = hidden[-1]
        comp = self.text2comp(text_embedding)
        fused = self.fusion(price_hidden, comp)
        return self.classifier(torch.cat([fused, comp], dim=-1))


