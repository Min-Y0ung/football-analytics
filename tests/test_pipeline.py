"""Pipeline tests on a small synthetic match (no downloaded data needed)."""
import numpy as np
import pandas as pd

from stfootball.baselines import constant_velocity
from stfootball.config import Config
from stfootball.db import connect, load_match
from stfootball.io.metrica import EVENT_COLS, Match
from stfootball.metrics import ade
from stfootball.preprocess import preprocess, remove_glitches, to_metres, unify_direction
from stfootball.windows import make_windows

CFG = Config()


def synthetic_match(n=25 * 10, glitch_at=None) -> Match:
    """22 players jogging in straight lines at 2 m/s, ball following home player 0.

    Period 1 has home on the left; period 2 swaps sides like a real match."""
    frame = np.arange(1, 2 * n + 1)
    period = np.repeat([1, 2], n)
    t = (frame - 1) / 25
    data = {"period": period, "frame": frame, "time": t}
    for side, base in (("H", 0.25), ("A", 0.75)):
        for i in range(11):
            x = np.where(period == 1, base, 1 - base) + 0.01 * (i - 5)
            x = x + (2.0 / 105) * (t - np.where(period == 1, 0, t[n]))
            y = 0.05 + 0.08 * i + np.zeros_like(t)
            data[f"{side}_P{i}_x"], data[f"{side}_P{i}_y"] = x, y
    data["ball_x"], data["ball_y"] = data["H_P0_x"], data["H_P0_y"]
    trk = pd.DataFrame(data)
    if glitch_at is not None:
        trk.loc[glitch_at:glitch_at + 5, "H_P3_x"] += 0.2  # ~21 m block jump for 6 frames, then back
    ev = pd.DataFrame([["H", "PASS", None, 1, 30, 40, "H_H_P0", "H_H_P1", 0.25, 0.05, 0.3, 0.1],
                       ["H", "PASS", None, 2, n + 30, n + 40, "H_H_P0", "H_H_P1", 0.75, 0.05, 0.7, 0.1]],
                      columns=EVENT_COLS)
    return Match("synthetic", trk, ev)


def test_to_metres_centre_origin_and_y_up():
    m = to_metres(pd.DataFrame({"period": [1], "ball_x": [1.0], "ball_y": [0.0]}))
    assert m.ball_x[0] == 52.5 and m.ball_y[0] == 34.0


def test_unify_direction_home_attacks_positive_x():
    trk = unify_direction(to_metres(synthetic_match().tracking))
    home = [c for c in trk.columns if c.startswith("H_") and c.endswith("_x")]
    for _, g in trk.groupby("period"):
        assert g[home].to_numpy().mean() < 0  # home defends the left half in both periods


def test_remove_glitches_restores_track():
    clean = to_metres(synthetic_match().tracking)
    dirty = to_metres(synthetic_match(glitch_at=100).tracking)
    fixed = remove_glitches(dirty, CFG)
    err = np.abs(fixed["H_P3_x"] - clean["H_P3_x"]).max()
    assert err < 0.5, err


def test_windows_shape_and_constant_velocity_is_exact_on_linear_motion():
    m = synthetic_match()
    w = make_windows(preprocess(m, CFG), m.match_id, CFG)
    t_in, t_out = int(CFG.input_s * CFG.target_fps), int(CFG.output_s * CFG.target_fps)
    assert w.X.shape[1:] == (t_in, 23, 4) and w.Y.shape[1:] == (t_out, 23, 2)
    assert len(w.X) > 0
    pred = constant_velocity(w.X, t_out, CFG.target_fps)
    assert ade(pred, w.Y) < 0.05


def test_db_roundtrip(tmp_path):
    m = synthetic_match()
    conn = connect(tmp_path / "t.db")
    load_match(conn, m, preprocess(m, CFG), CFG)
    load_match(conn, m, preprocess(m, CFG), CFG)  # reload replaces, never duplicates
    n_frames = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    n_pos = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    assert n_pos == 22 * n_frames
    # event coords follow the same direction unification: both passes start in home's left half
    xs = [r[0] for r in conn.execute("SELECT start_x FROM events ORDER BY period")]
    assert all(x < 0 for x in xs), xs


def test_downsample_gives_uniform_steps():
    from stfootball.preprocess import downsample
    out = downsample(to_metres(synthetic_match().tracking), CFG)
    for _, g in out.groupby("period"):
        assert np.allclose(np.diff(g["time"]), 1 / CFG.target_fps)
