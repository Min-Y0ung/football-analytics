"""Learned RQ1 models, leave-one-match-out, vs constant velocity.

    python scripts/train_baseline.py --model lstm
    python scripts/train_baseline.py --model transformer
    python scripts/train_baseline.py --model gcn_tcn --graph complete   # or knn / none

Writes ADE/FDE per match to runs/<name>_lomo.csv and the per-step error curve
to runs/<name>_horizon.npz (plot with scripts/plot_horizon.py), where <name> is the
model, plus the graph for gcn_tcn (e.g. gcn_tcn_knn).
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from stfootball.baselines import constant_velocity
from stfootball.config import DEFAULT
from stfootball.graphs import GRAPHS
from stfootball.io.metrica import load_all
from stfootball.metrics import ade, fde, horizon_error
from stfootball.models.gcn_tcn import GCNTCN
from stfootball.models.lstm import PlayerLSTM
from stfootball.models.transformer import PlayerTransformer
from stfootball.preprocess import preprocess
from stfootball.train import TrainConfig, fit, predict
from stfootball.windows import WindowSet, make_windows

MODELS = {"lstm": PlayerLSTM, "transformer": PlayerTransformer, "gcn_tcn": GCNTCN}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=MODELS, default="lstm")
ap.add_argument("--graph", choices=GRAPHS, default="complete", help="edges for gcn_tcn")
ap.add_argument("--data", default="data/metrica/data")
ap.add_argument("--cache", default="data/windows")
ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto")
ap.add_argument("--smooth-s", type=float, default=DEFAULT.smooth_window_s,
                help="centred smoothing window in seconds; 0 disables it (no future leakage)")
ap.add_argument("--out-dir", default="runs")
args = ap.parse_args()
cfg = replace(DEFAULT, smooth_window_s=args.smooth_s)
name = f"{args.model}_{args.graph}" if args.model == "gcn_tcn" else args.model


def build(t_out: int):
    kwargs = {"graph": args.graph} if args.model == "gcn_tcn" else {}
    return MODELS[args.model](t_out, **kwargs)


def load_windows() -> list[WindowSet]:
    cache = Path(args.cache) / f"smooth_{args.smooth_s:g}"
    cached = sorted(cache.glob("*.npz"))
    if cached:
        return [WindowSet(**{k: v for k, v in np.load(p).items()}, match_id=p.stem) for p in cached]
    cache.mkdir(parents=True, exist_ok=True)
    sets = []
    for m in load_all(args.data):
        w = make_windows(preprocess(m, cfg), m.match_id, cfg)
        np.savez_compressed(cache / f"{m.match_id}.npz", X=w.X, Y=w.Y, team=w.team, frame=w.frame)
        sets.append(w)
    return sets


sets = load_windows()
tc = TrainConfig(epochs=args.epochs, seed=args.seed, device=args.device)
t_out = int(cfg.output_s * cfg.target_fps)
rows, curves = [], {}
for test in sets:
    print(f"held out {test.match_id}", flush=True)
    model = fit(build(t_out), [w for w in sets if w is not test], tc, cfg)
    for label, pred in (("constant_velocity", constant_velocity(test.X, t_out, cfg.target_fps)),
                        (name, predict(model, test.X, device=args.device))):
        rows.append((test.match_id, label, len(test.X), ade(pred, test.Y), fde(pred, test.Y)))
        curves[f"{label}/{test.match_id}"] = horizon_error(pred, test.Y)

print(f"\n{'match':12s} {'model':18s} {'windows':>8s} {'ADE(m)':>7s} {'FDE(m)':>7s}")
for r in rows:
    print(f"{r[0]:12s} {r[1]:18s} {r[2]:8d} {r[3]:7.2f} {r[4]:7.2f}")
for m in ("constant_velocity", name):
    sel = [r for r in rows if r[1] == m]
    print(f"{'mean':12s} {m:18s} {sum(r[2] for r in sel):8d} "
          f"{np.mean([r[3] for r in sel]):7.2f} {np.mean([r[4] for r in sel]):7.2f}")

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
with open(out / f"{name}_lomo.csv", "w", newline="") as f:
    csv.writer(f).writerows([("match", "model", "windows", "ade", "fde"), *rows])
np.savez(out / f"{name}_horizon.npz", **curves)
print(f"saved {out}/{name}_lomo.csv, {out}/{name}_horizon.npz")
