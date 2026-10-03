"""Training loop shared by the learned RQ1 models (players only, ball is input-free here).

Validation windows are the last `val_frac` of each training match, separated from the
training part by a gap so overlapping windows never sit on both sides of the split.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import DEFAULT, Config
from .windows import WindowSet

PLAYERS = slice(0, 22)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    val_frac: float = 0.1
    flip_y: bool = True   # mirror augmentation (y -> -y); keeps home attacking +x
    seed: int = 0
    device: str = "auto"


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_train_val(sets: list[WindowSet], val_frac: float, cfg: Config = DEFAULT):
    gap = math.ceil((cfg.input_s + cfg.output_s) / cfg.stride_s) - 1
    tr, va = [], []
    for w in sets:
        n_val = max(1, int(len(w.X) * val_frac))
        cut = len(w.X) - n_val
        tr.append((w.X[:max(0, cut - gap)], w.Y[:max(0, cut - gap)]))
        va.append((w.X[cut:], w.Y[cut:]))
    cat = lambda parts, i: np.concatenate([p[i] for p in parts])
    return (cat(tr, 0), cat(tr, 1)), (cat(va, 0), cat(va, 1))


def displacement_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean error, i.e. ADE as a differentiable loss."""
    return (pred - true).norm(dim=-1).mean()


def _flip(X: torch.Tensor, Y: torch.Tensor):
    X, Y = X.clone(), Y.clone()
    X[..., 1] *= -1; X[..., 3] *= -1; Y[..., 1] *= -1
    return X, Y


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, batch_size: int = 256, device: str = "auto") -> np.ndarray:
    dev = pick_device(device)
    model.to(dev).eval()
    out = [model(torch.as_tensor(X[i:i + batch_size, :, PLAYERS], dtype=torch.float32, device=dev)).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def fit(model: torch.nn.Module, train_sets: list[WindowSet], tc: TrainConfig = TrainConfig(),
        cfg: Config = DEFAULT, log=print) -> torch.nn.Module:
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    dev = pick_device(tc.device)
    (Xtr, Ytr), (Xva, Yva) = split_train_val(train_sets, tc.val_frac, cfg)
    Xtr = torch.as_tensor(Xtr[:, :, PLAYERS], dtype=torch.float32)
    Ytr = torch.as_tensor(Ytr[:, :, PLAYERS], dtype=torch.float32)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, tc.epochs + 1):
        model.train()
        order = rng.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(order), tc.batch_size):
            idx = order[i:i + tc.batch_size]
            X, Y = Xtr[idx], Ytr[idx]
            if tc.flip_y and rng.random() < 0.5:
                X, Y = _flip(X, Y)
            loss = displacement_loss(model(X.to(dev)), Y.to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        val = float(np.linalg.norm(predict(model, Xva, device=tc.device) - Yva[:, :, PLAYERS], axis=-1).mean())
        sched.step(val)
        log(f"  epoch {epoch:3d}  train ADE {total / len(Xtr):.3f}  val ADE {val:.3f}")
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc.patience:
                break
    model.load_state_dict(best_state)
    return model
