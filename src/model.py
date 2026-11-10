import torch
import torch.nn as nn

class ProxyControlledPolicy(nn.Module):
    """Policy head with proxy controls for latent confounders."""

    def __init__(self, context_dim: int, action_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.head = nn.Linear(64, action_dim)

    def forward(self, context):
        return torch.softmax(self.head(self.encoder(context)), dim=-1)
