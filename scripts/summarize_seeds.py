"""Mean +- std over seeds of the leave-one-match-out ADE/FDE in runs/.

Seed 0 is runs/<name>_lomo.csv, other seeds runs/seeds/seed<k>/<name>_lomo.csv, e.g.

    python scripts/train_baseline.py --model gcn_tcn --graph complete --seed 1 --out-dir runs/seeds/seed1
    python scripts/summarize_seeds.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
args = ap.parse_args()
runs = Path(args.runs)

print(f"{'model':18s} {'seeds':>5s} {'ADE(m)':>15s} {'FDE(m)':>15s}")
for f in sorted(runs.glob("*_lomo.csv")):
    name = f.name.removesuffix("_lomo.csv")
    files = [f, *sorted(runs.glob(f"seeds/seed*/{name}_lomo.csv"))]
    per_seed = []  # (ADE, FDE) averaged over held-out matches
    for path in files:
        rows = [r for r in csv.DictReader(open(path)) if r["model"] == name]
        per_seed.append((np.mean([float(r["ade"]) for r in rows]), np.mean([float(r["fde"]) for r in rows])))
    a, d = np.array(per_seed).T
    sd = (lambda x: x.std(ddof=1)) if len(files) > 1 else (lambda x: 0.0)
    print(f"{name:18s} {len(files):5d} {a.mean():8.3f} ± {sd(a):.3f} {d.mean():8.3f} ± {sd(d):.3f}")
