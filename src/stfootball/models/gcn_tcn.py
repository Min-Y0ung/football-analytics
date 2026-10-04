"""GCN + TCN spatio-temporal graph model for RQ1.

Each block mixes information over time (dilated temporal convolution, per node)
and then over the player graph (dense GCN from PyTorch Geometric, per time step).
A node's own features go through a separate root weight instead of a GCN self-loop:
with self-loops a complete graph averages each player with 22 others at weight 1/23
and the model underfits (train ADE 0.60 vs 0.47 without edges). With graph="none"
only the root path is left, which gives the same network without interactions,
i.e. the ablation for the graph.
"""
import torch
from torch import nn
from torch_geometric.nn import DenseGCNConv

from ..graphs import adjacency
from .lstm import POS_SCALE, node_features

N_NODES = 23
TEAM = torch.tensor([0] * 11 + [1] * 11 + [2])  # home, away, ball (windows.py node order)


class STBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.temporal = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.neighbours = DenseGCNConv(ch, ch)
        self.root = nn.Linear(ch, ch, bias=False)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, T, N, C = h.shape
        t = self.temporal(h.permute(0, 2, 3, 1).reshape(B * N, C, T))
        t = torch.relu(t).view(B, N, C, T).permute(0, 3, 1, 2)
        s = self.neighbours(t.reshape(B * T, N, C), adj.reshape(B * T, N, N), add_loop=False).view(B, T, N, C)
        s = s + self.root(t)
        return self.norm(h + self.drop(torch.relu(s)))


class GCNTCN(nn.Module):
    uses_ball = True  # the ball node is part of the graph; outputs cover the 22 players

    def __init__(self, t_out: int, graph: str = "complete", hidden: int = 64,
                 dilations=(1, 2, 4, 8), k: int = 4, dropout: float = 0.1):
        super().__init__()
        self.t_out, self.graph, self.k = t_out, graph, k
        self.embed = nn.Linear(6 + 3, hidden)
        self.blocks = nn.ModuleList(STBlock(hidden, d, dropout) for d in dilations)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, t_out * 2))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(B, T_in, 23, 4) -> (B, T_out, 22, 2) future player positions in metres."""
        B, T, N, _ = X.shape
        team = nn.functional.one_hot(TEAM.to(X.device), 3).float().expand(B, T, N, 3)
        h = self.embed(torch.cat([node_features(X), team], -1))
        adj = adjacency(X[..., :2], self.graph, self.k)
        for block in self.blocks:
            h = block(h, adj)
        offset = self.head(h[:, -1, :22]).view(B, 22, self.t_out, 2).permute(0, 2, 1, 3) * POS_SCALE
        return X[:, -1:, :22, :2] + offset
