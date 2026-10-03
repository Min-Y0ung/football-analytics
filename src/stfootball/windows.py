"""Sliding windows for trajectory prediction (RQ1).

A window is kept only if the ball and exactly 11 players per team are tracked on
every frame and the on-pitch line-up does not change inside it. Node order is
fixed per window: home players then away players, each sorted by mean x at the
last observed frame (defence -> attack), then the ball as node 22.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT, Config


@dataclass
class WindowSet:
    X: np.ndarray        # (N, T_in, 23, 4)  x, y, vx, vy
    Y: np.ndarray        # (N, T_out, 23, 2) future x, y
    team: np.ndarray     # (N, 23) 0 = home, 1 = away, 2 = ball
    frame: np.ndarray    # (N,) raw frame id of the last observed step
    match_id: str


def make_windows(trk: pd.DataFrame, match_id: str, cfg: Config = DEFAULT) -> WindowSet:
    fps = cfg.target_fps
    t_in, t_out, stride = int(cfg.input_s * fps), int(cfg.output_s * fps), int(cfg.stride_s * fps)
    span = t_in + t_out
    players = sorted({c[:-2] for c in trk.columns if c.endswith("_x") and c != "ball_x"})
    P = np.stack([trk[[f"{p}_x", f"{p}_y", f"{p}_vx", f"{p}_vy"]].to_numpy() for p in players], 1)
    B = trk[["ball_x", "ball_y", "ball_vx", "ball_vy"]].to_numpy()
    on = ~np.isnan(P).any(-1)
    is_home = np.array([p.startswith("H_") for p in players])
    period = trk["period"].to_numpy()
    frames = trk["frame"].to_numpy()

    Xs, Ys, F = [], [], []
    for s in range(0, len(trk) - span + 1, stride):
        e = s + span
        if period[s] != period[e - 1] or np.isnan(B[s:e]).any():
            continue
        lineup = on[s]
        if not (on[s:e] == lineup).all() or lineup[is_home].sum() != 11 or lineup[~is_home].sum() != 11:
            continue
        last = s + t_in - 1
        h = np.where(lineup & is_home)[0]
        a = np.where(lineup & ~is_home)[0]
        h = h[np.argsort(P[last, h, 0])]
        a = a[np.argsort(-P[last, a, 0])]  # away attacks towards -x
        seq = np.concatenate([P[s:e][:, np.r_[h, a]], B[s:e, None]], axis=1)
        Xs.append(seq[:t_in]); Ys.append(seq[t_in:, :, :2]); F.append(frames[last])

    team = np.array([0] * 11 + [1] * 11 + [2])
    if not Xs:
        return WindowSet(np.empty((0, t_in, 23, 4)), np.empty((0, t_out, 23, 2)), team[None], np.empty(0, int), match_id)
    return WindowSet(np.stack(Xs), np.stack(Ys), np.tile(team, (len(Xs), 1)), np.array(F), match_id)
