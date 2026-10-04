"""Per-player Transformer baseline for RQ1.

Same setting as `PlayerLSTM` (each player alone, shared weights) with self-attention
over the 4 s history instead of recurrence, so the two isolate the effect of the
temporal model.
"""
import torch
from torch import nn

from .lstm import POS_SCALE, node_features


class PlayerTransformer(nn.Module):
    def __init__(self, t_out: int, t_in: int = 40, d_model: int = 128, heads: int = 4,
                 layers: int = 3, ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.t_out = t_out
        self.embed = nn.Linear(6, d_model)
        self.pos = nn.Parameter(torch.zeros(1, t_in, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, t_out * 2))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(B, T_in, N, 4) -> (B, T_out, N, 2) future positions in metres."""
        B, T, N, _ = X.shape
        f = node_features(X).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h = self.norm(self.encoder(self.embed(f) + self.pos[:, :T]))
        offset = self.head(h[:, -1]).view(B, N, self.t_out, 2).permute(0, 2, 1, 3) * POS_SCALE
        return X[:, -1:, :, :2] + offset
