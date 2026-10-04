"""Error vs prediction horizon for every model in runs/*_horizon.npz (mean over held-out matches).

    python scripts/plot_horizon.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stfootball.config import DEFAULT as cfg

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="docs/figures/fig3_horizon_error.png")
args = ap.parse_args()

by_key = {}  # "model/match" -> curve; constant velocity repeats across files, keep one copy
for f in sorted(Path(args.runs).glob("*_horizon.npz")):
    by_key.update(np.load(f))
curves: dict[str, list[np.ndarray]] = {}
for key, curve in by_key.items():
    curves.setdefault(key.split("/")[0], []).append(curve)

t = np.arange(1, int(cfg.output_s * cfg.target_fps) + 1) / cfg.target_fps
fig, ax = plt.subplots(figsize=(6, 4))
for model, cs in curves.items():
    mean = np.mean(cs, axis=0)
    ax.plot(t, mean, marker="o", ms=3, label=f"{model} (ADE {mean.mean():.2f}, FDE {mean[-1]:.2f} m)")
ax.set_xlabel("prediction horizon (s)")
ax.set_ylabel("mean displacement error (m)")
ax.set_title("RQ1: error vs horizon (leave-one-match-out, players only)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150)
print(f"saved {args.out}")
