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
  -> home always attacks +x  ->  glitch removal (> 12 m/s)  ->  0.5 s smoothing
  -> uniform 10 Hz resampling  ->  velocities
  -> SQLite (matches / players / frames / positions / events)
  -> 6 s windows (4 s in, 2 s out), 23 nodes (11 + 11 + ball)  ->  models
```

## Quick start

```bash
pip install -e .            # or: pip install -r requirements.txt
bash scripts/download_metrica.sh          # 3 open matches -> data/metrica
python scripts/build_db.py                # -> data/football.db (~170k frames, 3.8M positions)
python scripts/run_baselines.py           # RQ1 baselines
pytest                                    # synthetic-data tests, no download needed
```

## Results so far (RQ1 baselines, 2 s horizon, players only)

| Match | Windows | Stationary ADE / FDE | Constant velocity ADE / FDE |
|---|---|---|---|
| metrica_1 | 2,730 | 2.25 / 4.21 m | 0.71 / 1.84 m |
| metrica_2 | 2,500 | 2.28 / 4.27 m | 0.73 / 1.90 m |
| metrica_3 | 3,071 | 2.33 / 4.35 m | 0.80 / 2.07 m |

Learned models will be evaluated leave-one-match-out.

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
  metrics.py      ADE, FDE
scripts/          download, build_db, run_baselines
tests/            pipeline tests on a synthetic match
docs/             data notes and figures
```

## Roadmap

- [x] Data loaders, cleaning, SQLite store, baselines
- [ ] LSTM / Transformer per-player baselines
- [ ] Graph construction (complete / kNN / pass-availability edges) + GCN-TCN in PyTorch Geometric
- [ ] Pitch control (Spearman 2018) and location value
- [ ] Virtual tactical simulation and Space Creation Value
- [ ] Extension: graph Neural ODE for continuous-time trajectories

## Acknowledgements

Tracking and event data by [Metrica Sports](https://metrica-sports.com/). Advisor: 전진성 교수님 (Korea University Sejong).
