"""Per-player LSTM baseline for RQ1.

Every player is encoded as an independent sequence with shared weights, so the model
sees no interactions. This is the reference the graph models (GCN + TCN) must beat.
"""
import torch
from torch import nn

POS_SCALE = 10.0   # metres; displacements within a 4 s window are of this order
VEL_SCALE = 5.0    # m/s
HALF_PITCH = (52.5, 34.0)


def node_features(X: torch.Tensor) -> torch.Tensor:
    """(B, T, N, 4) x, y, vx, vy -> (B, T, N, 6) scaled features.

    Positions are taken relative to the last observed frame (translation invariant);
    absolute pitch position is kept separately so the model can learn touchline effects."""
    pos, vel = X[..., :2], X[..., 2:]
    rel = (pos - pos[:, -1:]) / POS_SCALE
    absolute = pos / pos.new_tensor(HALF_PITCH)
    return torch.cat([rel, vel / VEL_SCALE, absolute], -1)


class PlayerLSTM(nn.Module):
    def __init__(self, t_out: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.t_out = t_out
        self.embed = nn.Linear(6, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, t_out * 2))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(B, T_in, N, 4) -> (B, T_out, N, 2) future positions in metres."""
        B, T, N, _ = X.shape
        f = node_features(X).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h, _ = self.lstm(torch.relu(self.embed(f)))
        offset = self.head(h[:, -1]).view(B, N, self.t_out, 2).permute(0, 2, 1, 3) * POS_SCALE
        return X[:, -1:, :, :2] + offset
