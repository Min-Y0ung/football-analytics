# CLAUDE.md

Research code for an undergraduate project (S-CURT, Korea University Sejong, 2026-09-23 ~ 2026-12-18):
ST-GNN trajectory prediction for football tracking data (RQ1) and Space Creation Value via
counterfactual "virtual tactical simulation" (RQ2). Reply to the owner in Korean.

## Commands
- `pip install -e '.[ml]'` (torch + PyG) then `bash scripts/download_metrica.sh` (data goes to `data/`, gitignored)
- `python scripts/build_db.py` builds `data/football.db` (SQLite)
- `python scripts/run_baselines.py` prints ADE/FDE per match
- `python scripts/train_lstm.py` trains the per-player LSTM leave-one-match-out (windows cached in `data/windows/`)
- `pytest` runs synthetic-data tests (no download needed); run before every commit

## Conventions
- Coordinates: metres, origin at centre spot, x towards the right goal, y up; home always attacks +x.
- No centred smoothing on model inputs (it leaks future frames). Tracking is resampled to a uniform 10 Hz grid; `frame` keeps the nearest raw 25 Hz frame id for joining events.
- Windows: 4 s in -> 2 s out, 23 nodes (11 home, 11 away, ball), node order fixed per window in `windows.py`.
- Evaluate learned models leave-one-match-out; never split overlapping windows at frame level.
- Scope promised in the program application: Metrica data + RDB pipeline, player graph with
  interaction / pass-availability edges, GCN + TCN in PyTorch Geometric, trajectory error +
  virtual tactical simulation, a clean GitHub repo. Neural ODE is an extension after the program.
- Keep personal documents (application, schedule) out of this public repo.

## Next steps
1. Transformer per-player baseline (LSTM done: ADE/FDE 0.48/1.30 m vs constant velocity 0.76/1.97 m)
2. Graph construction (complete / kNN / pass-availability) + GCN-TCN
3. Pitch control (Spearman 2018) + location value grid -> Space Creation Value
