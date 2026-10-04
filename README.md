# football-analytics

**Spatio-temporal GNN trajectory prediction and space creation value for football tracking data**

> 시공간 그래프 신경망(ST-GNN)으로 축구 선수 22명의 궤적을 예측하고, 오프더볼 움직임이 만든 공간의 가치를 정량화하는 연구 (고려대학교 세종캠퍼스 S-CURT 2026-2)

## Research questions

| | Question | Metric |
|---|---|---|
| **RQ1** | Given 4 s of all 22 players + ball, how well can we predict the next 2 s? Does modelling player interactions as a graph (GCN + TCN) beat per-player baselines? | ADE / FDE (m) |
| **RQ2** | How much space did a player's off-ball run create? Replace their real trajectory with the model's "typical" prediction (a virtual tactical simulation) and measure the change in pitch control weighted by location value. | Space Creation Value |

The RQ1 model doubles as the counterfactual baseline for RQ2.

## Pipeline

```
Metrica raw (25 Hz, normalized)  ->  metres, centre origin, y up
  -> home always attacks +x  ->  glitch removal (> 12 m/s)  ->  (optional smoothing, off)
  -> uniform 10 Hz resampling  ->  velocities
  -> SQLite (matches / players / frames / positions / events)
  -> 6 s windows (4 s in, 2 s out), 23 nodes (11 + 11 + ball)  ->  models
```

## Quick start

```bash
pip install -e '.[ml]'      # or: pip install -r requirements.txt  (drop [ml] for data only)
bash scripts/download_metrica.sh          # 3 open matches -> data/metrica
python scripts/build_db.py                # -> data/football.db (~170k frames, 3.8M positions)
python scripts/run_baselines.py           # RQ1 baselines
python scripts/train_baseline.py --model lstm         # per-player LSTM, leave-one-match-out (~25 min on Apple MPS)
python scripts/train_baseline.py --model transformer  # per-player Transformer (~45 min)
python scripts/train_baseline.py --model gcn_tcn --graph complete  # GCN-TCN; also knn / none (~25 min each)
python scripts/plot_horizon.py                        # error vs horizon figure from runs/
pytest                                    # synthetic-data tests, no download needed
```

## Results so far (RQ1, 2 s horizon, players only, leave-one-match-out)

ADE / FDE in metres per held-out match.

| Model | Interactions | metrica_1 | metrica_2 | metrica_3 | **mean** |
|---|---|---|---|---|---|
| Stationary | - | 2.26 / 4.22 | 2.29 / 4.28 | 2.33 / 4.36 | 2.29 / 4.29 |
| Constant velocity | - | 0.72 / 1.86 | 0.75 / 1.93 | 0.82 / 2.11 | 0.76 / 1.97 |
| Per-player LSTM | no | 0.43 / 1.19 | 0.45 / 1.24 | 0.55 / 1.48 | **0.47** / 1.30 |
| Per-player Transformer | no | 0.47 / 1.28 | 0.48 / 1.30 | 0.57 / 1.51 | 0.50 / 1.36 |
| GCN-TCN, no edges | no | 0.46 / 1.26 | 0.48 / 1.32 | 0.58 / 1.52 | 0.51 / 1.37 |
| GCN-TCN, kNN (k = 4) | yes | 0.44 / 1.21 | 0.46 / 1.24 | 0.56 / 1.47 | 0.49 / 1.31 |
| GCN-TCN, complete | yes | 0.43 / 1.17 | 0.45 / 1.22 | 0.55 / 1.44 | 0.48 / **1.27** |

Windows per match: 2,733 / 2,483 / 3,059. Learned models: `python scripts/train_baseline.py --model lstm|transformer|gcn_tcn [--graph none|knn|complete]`,
up to 80 epochs with early stopping, one seed.

- **Interactions help.** The GCN-TCN with edges beats the same network without edges in every held-out match
  (complete: ADE -0.03 m, FDE -0.10 m). This is the RQ1 comparison.
- Against the per-player LSTM, the complete-graph GCN-TCN ties on ADE and is better on FDE in every match,
  with a quarter of the parameters (74k vs 287k).
- Swapping recurrence for attention (LSTM -> Transformer) does not help on 4 s histories.
- Smoothing is off: a centred 0.5 s moving average leaked future frames into the inputs and made the LSTM look better than it is (0.36 / 1.07 m).

![error vs horizon](docs/figures/fig3_horizon_error.png)

## Data

[Metrica Sports sample data](https://github.com/metrica-sports/sample-data): 3 anonymized matches with synchronized tracking (25 Hz) and events. Data are downloaded, not committed. Findings from the exploratory analysis are in [docs/data.md](docs/data.md).

![snapshot](docs/figures/fig1_snapshot.png)

## Layout

```
src/stfootball/
  io/metrica.py   loaders for CSV (games 1-2) and FIFA EPTS (game 3, substitution-aware)
  preprocess.py   metres, direction, glitch removal, smoothing, 10 Hz resampling, velocity
  db.py           SQLite schema, loader, event-window queries
  windows.py      fixed-size 23-node sliding windows
  baselines.py    stationary, constant velocity
  graphs.py       per-step adjacency: none / complete / kNN
  models/         per-player LSTM and Transformer baselines, GCN-TCN
  train.py        training loop, train/val split with overlap gap
  metrics.py      ADE, FDE, error per horizon step
scripts/          download, build_db, run_baselines, train_baseline, plot_horizon
tests/            pipeline and model tests on synthetic data
docs/             data notes and figures
```

## Roadmap

- [x] Data loaders, cleaning, SQLite store, baselines
- [x] Per-player LSTM baseline
- [x] Transformer per-player baseline
- [x] Graph construction (complete / kNN) + GCN-TCN in PyTorch Geometric
- [ ] Pass-availability edges, larger GCN-TCN, multiple seeds
- [ ] Pitch control (Spearman 2018) and location value
- [ ] Virtual tactical simulation and Space Creation Value
- [ ] Extension: graph Neural ODE for continuous-time trajectories

## Acknowledgements

Tracking and event data by [Metrica Sports](https://metrica-sports.com/). Advisor: 전진성 교수님 (Korea University Sejong).
