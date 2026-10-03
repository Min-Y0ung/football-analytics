"""RQ1 baselines with leave-one-match-out evaluation (no training needed for these two).

    python scripts/run_baselines.py --data data/metrica/data
"""
import argparse

import numpy as np

from stfootball.baselines import constant_velocity, stationary
from stfootball.config import DEFAULT as cfg
from stfootball.io.metrica import load_all
from stfootball.metrics import ade, fde
from stfootball.preprocess import preprocess
from stfootball.windows import make_windows

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/metrica/data")
args = ap.parse_args()

t_out = int(cfg.output_s * cfg.target_fps)
rows = []
for m in load_all(args.data):
    w = make_windows(preprocess(m), m.match_id)
    for name, pred in (("stationary", stationary(w.X, t_out)),
                       ("constant_velocity", constant_velocity(w.X, t_out, cfg.target_fps))):
        rows.append((m.match_id, name, len(w.X), ade(pred, w.Y), fde(pred, w.Y)))

print(f"{'match':12s} {'model':18s} {'windows':>8s} {'ADE(m)':>7s} {'FDE(m)':>7s}")
for r in rows:
    print(f"{r[0]:12s} {r[1]:18s} {r[2]:8d} {r[3]:7.2f} {r[4]:7.2f}")
