"""Player graphs for the ST-GNN (RQ1).

Adjacency is built per time step from node positions, so edges follow the play
inside a window. Nodes are the 23 of `windows.py` (11 home, 11 away, ball).
Self-loops are left out; the GCN layer adds them.
"""
import torch

GRAPHS = ("none", "complete", "knn")


def adjacency(pos: torch.Tensor, kind: str, k: int = 4) -> torch.Tensor:
    """(..., N, 2) positions -> (..., N, N) 0/1 adjacency without self-loops.

    none      no edges: every node is processed alone (same model, interactions ablated)
    complete  every pair of nodes
    knn       each node to its k nearest nodes, symmetrised
    """
    N = pos.shape[-2]
    eye = torch.eye(N, dtype=torch.bool, device=pos.device)
    if kind == "none":
        return torch.zeros(*pos.shape[:-1], N, device=pos.device)
    if kind == "complete":
        return (~eye).float().expand(*pos.shape[:-1], N).clone()
    if kind == "knn":
        d = torch.cdist(pos, pos).masked_fill(eye, float("inf"))
        idx = d.topk(k, dim=-1, largest=False).indices
        A = torch.zeros_like(d).scatter_(-1, idx, 1.0)
        return torch.maximum(A, A.transpose(-1, -2))
    raise ValueError(f"unknown graph {kind!r}, expected one of {GRAPHS}")
