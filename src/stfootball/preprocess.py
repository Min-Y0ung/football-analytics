"""Cleaning pipeline: normalized Metrica coords -> clean, direction-unified metres at 10 Hz.

Coordinate convention after ``to_metres``: origin at the centre spot, x in
[-52.5, 52.5] towards the right goal, y in [-34, 34] pointing up (Metrica's y
axis points down, so it is flipped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT, Config
from .io.metrica import Match


def xy_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    xs = [c for c in df.columns if c.endswith("_x")]
    return xs, [c[:-2] + "_y" for c in xs]


def to_metres(trk: pd.DataFrame, cfg: Config = DEFAULT) -> pd.DataFrame:
    out = trk.copy()
    xs, ys = xy_columns(out)
    out[xs] = (out[xs] - 0.5) * cfg.pitch_length
    out[ys] = -(out[ys] - 0.5) * cfg.pitch_width
    return out


def flipped_periods(trk: pd.DataFrame, centre: float = 0.0) -> set[int]:
    """Periods in which the home team sits in the right half (attacks towards -x).

    ``centre`` is the halfway-line x: 0.0 for metres (centre origin), 0.5 for raw Metrica coords."""
    xs, _ = xy_columns(trk)
    home_x = [c for c in xs if c.startswith("H_")]
    return {int(p) for p, g in trk.groupby("period") if np.nanmean(g[home_x].to_numpy()) > centre}


def unify_direction(trk: pd.DataFrame) -> pd.DataFrame:
    """Rotate periods by 180 degrees so the home team always attacks towards +x."""
    out = trk.copy()
    xs, ys = xy_columns(out)
    flip = out["period"].isin(flipped_periods(out))
    out.loc[flip, xs + ys] = -out.loc[flip, xs + ys]
    return out


def _speed_gate(x: np.ndarray, y: np.ndarray, max_step: float) -> np.ndarray:
    """Forward pass that rejects points unreachable from the last accepted point.

    A point k frames after the last accepted one is accepted only if it lies within
    ``k * max_step`` metres. This removes both ramps and block-shifted segments
    (a jump away and back), which a plain per-step speed threshold misses."""
    bad = np.zeros(len(x), bool)
    ax = ay = None
    last = 0
    for i in range(len(x)):
        if np.isnan(x[i]):
            ax = None  # off the pitch (substitution): restart from the next seen point
            continue
        if ax is None or np.hypot(x[i] - ax, y[i] - ay) <= (i - last) * max_step:
            ax, ay, last = x[i], y[i], i
        else:
            bad[i] = True
    return bad


def remove_glitches(trk: pd.DataFrame, cfg: Config = DEFAULT) -> pd.DataFrame:
    """Blank physically impossible player positions (> ``cfg.max_speed``) and interpolate.

    Metrica's raw feed contains short position jumps of 13-60 m/s. Only interior
    gaps up to 1 s are filled, so substitutions (long NaN runs) stay NaN."""
    out = trk.copy()
    xs, _ = xy_columns(out)
    players = [c[:-2] for c in xs if c != "ball_x"]
    max_step = cfg.max_speed / cfg.raw_fps
    fill = lambda a: pd.Series(a).interpolate(limit=cfg.raw_fps, limit_area="inside").to_numpy()
    for _, idx in out.groupby("period").groups.items():
        for p in players:
            x = out.loc[idx, f"{p}_x"].to_numpy().copy()
            y = out.loc[idx, f"{p}_y"].to_numpy().copy()
            bad = _speed_gate(x, y, max_step)
            if bad.any():
                x[bad], y[bad] = np.nan, np.nan
                out.loc[idx, f"{p}_x"], out.loc[idx, f"{p}_y"] = fill(x), fill(y)
    return out


def smooth(trk: pd.DataFrame, cfg: Config = DEFAULT) -> pd.DataFrame:
    """Centered moving average per period (players only; the ball moves too fast to smooth).

    Uses future frames, so keep it off for prediction inputs (window 0 = no-op)."""
    out = trk.copy()
    xs, ys = xy_columns(out)
    cols = [c for c in xs + ys if not c.startswith("ball")]
    win = max(1, int(round(cfg.smooth_window_s * cfg.raw_fps)) | 1)
    out[cols] = out.groupby("period")[cols].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    return out


def downsample(trk: pd.DataFrame, cfg: Config = DEFAULT) -> pd.DataFrame:
    """Resample each period onto a uniform ``target_fps`` grid by linear interpolation.

    25 -> 10 Hz is not an integer step, so picking raw frames would give uneven
    0.08/0.12 s steps and wrong velocities. A value is NaN if either neighbour is NaN.
    ``frame`` keeps the nearest raw frame id so events can still be joined."""
    xs, ys = xy_columns(trk)
    cols = xs + ys
    parts = []
    for period, g in trk.groupby("period"):
        t = g["time"].to_numpy()
        grid = np.arange(t[0], t[-1] + 1e-9, 1 / cfg.target_fps)
        r = np.searchsorted(t, grid).clip(1, len(t) - 1)
        l = r - 1
        w = ((grid - t[l]) / (t[r] - t[l]))[:, None]
        v = g[cols].to_numpy()
        res = pd.DataFrame(v[l] * (1 - w) + v[r] * w, columns=cols)
        nearest = np.where(w[:, 0] < 0.5, l, r)
        res.insert(0, "time", grid)
        res.insert(0, "frame", g["frame"].to_numpy()[nearest])
        res.insert(0, "period", period)
        parts.append(res)
    return pd.concat(parts, ignore_index=True)


def add_velocity(trk: pd.DataFrame, fps: float) -> pd.DataFrame:
    out = trk.copy()
    xs, ys = xy_columns(out)
    vel = {}
    for c in xs + ys:
        vel[c[:-2] + "_v" + c[-1]] = out.groupby("period")[c].diff() * fps
    return pd.concat([out, pd.DataFrame(vel, index=out.index)], axis=1)


def preprocess(match: Match, cfg: Config = DEFAULT) -> pd.DataFrame:
    """Full pipeline used for modelling. Returns 10 Hz tracking in metres with velocities."""
    trk = to_metres(match.tracking, cfg)
    trk = unify_direction(trk)
    trk = remove_glitches(trk, cfg)
    trk = smooth(trk, cfg)
    trk = downsample(trk, cfg)
    return add_velocity(trk, cfg.target_fps)
