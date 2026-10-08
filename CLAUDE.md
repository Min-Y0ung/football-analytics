# CLAUDE.md

Research code for an undergraduate project (S-CURT, Korea University Sejong, 2026-09-23 ~ 2026-12-18):
ST-GNN trajectory prediction for football tracking data (RQ1) and Space Creation Value via
counterfactual "virtual tactical simulation" (RQ2). Reply to the owner in Korean.

## Commands
- `pip install -e '.[ml]'` (torch + PyG) then `bash scripts/download_metrica.sh` (data goes to `data/`, gitignored)
- `python scripts/build_db.py` builds `data/football.db` (SQLite)
- `python scripts/run_baselines.py` prints ADE/FDE per match
- `python scripts/train_baseline.py --model lstm|transformer|gcn_tcn [--graph none|knn|complete] [--seed k --out-dir runs/seeds/seedk]` trains RQ1 models leave-one-match-out (windows cached in `data/windows/`); `python scripts/plot_horizon.py` plots error vs horizon; `python scripts/summarize_seeds.py` gives mean ± std over seeds
- `pytest` runs synthetic-data tests (no download needed); run before every commit

## Conventions
- Write PR descriptions with a Korean section first, English below.
- Coordinates: metres, origin at centre spot, x towards the right goal, y up; home always attacks +x.
- No centred smoothing on model inputs (it leaks future frames). Tracking is resampled to a uniform 10 Hz grid; `frame` keeps the nearest raw 25 Hz frame id for joining events.
- Windows: 4 s in -> 2 s out, 23 nodes (11 home, 11 away, ball), node order fixed per window in `windows.py`.
- Evaluate learned models leave-one-match-out; never split overlapping windows at frame level.
- Scope promised in the program application: Metrica data + RDB pipeline, player graph with
  interaction / pass-availability edges, GCN + TCN in PyTorch Geometric, trajectory error +
  virtual tactical simulation, a clean GitHub repo. Neural ODE is an extension after the program.
- Keep personal documents (application, schedule) out of this public repo.

## Next steps
Current ADE/FDE (mean of seeds 0-2): constant velocity 0.76/1.97 m, per-player LSTM 0.474/1.298 m,
Transformer 0.50/1.36 m (seed 0 only), GCN-TCN none/kNN/complete 0.509/1.374, 0.491/1.313, 0.482/1.284 m.
Edges beat no edges in all 9 seed x match pairs; complete GCN-TCN vs LSTM is a tie (within 0.02 m). GCN uses a separate root weight; with
GCN self-loops the complete graph underfit (0.64/1.60 m).
1. Pass-availability edges; scale GCN-TCN up (report 3 seeds for new models)
2. Pitch control (Spearman 2018) + location value grid -> Space Creation Value
