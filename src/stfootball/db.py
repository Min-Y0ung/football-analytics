"""Relational store for tracking + event data (SQLite).

Positions are stored in long format (one row per player per frame) with a
composite index, so time windows and per-player trajectories are plain SQL.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DEFAULT, Config
from .io.metrica import Match
from .preprocess import flipped_periods

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id     TEXT PRIMARY KEY,
    provider     TEXT,
    fps          REAL,
    pitch_length REAL,
    pitch_width  REAL
);
CREATE TABLE IF NOT EXISTS players (
    match_id  TEXT REFERENCES matches(match_id),
    player_id TEXT,
    team      TEXT CHECK (team IN ('H', 'A')),
    PRIMARY KEY (match_id, player_id)
);
CREATE TABLE IF NOT EXISTS frames (
    match_id TEXT REFERENCES matches(match_id),
    frame_id INTEGER,
    period   INTEGER,
    time_s   REAL,
    ball_x   REAL, ball_y REAL, ball_vx REAL, ball_vy REAL,
    PRIMARY KEY (match_id, frame_id)
);
CREATE TABLE IF NOT EXISTS positions (
    match_id  TEXT,
    frame_id  INTEGER,
    player_id TEXT,
    x REAL, y REAL, vx REAL, vy REAL,
    PRIMARY KEY (match_id, frame_id, player_id),
    FOREIGN KEY (match_id, frame_id) REFERENCES frames(match_id, frame_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id    TEXT REFERENCES matches(match_id),
    team        TEXT,
    type        TEXT,
    subtype     TEXT,
    period      INTEGER,
    start_frame INTEGER,
    end_frame   INTEGER,
    from_player TEXT,
    to_player   TEXT,
    start_x REAL, start_y REAL, end_x REAL, end_y REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_player ON positions (match_id, player_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_events_frame ON events (match_id, start_frame);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (match_id, type);
"""


def connect(path: str | Path = "data/football.db") -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def load_match(conn: sqlite3.Connection, match: Match, trk: pd.DataFrame, cfg: Config = DEFAULT) -> None:
    """Insert one match. ``trk`` is the preprocessed (metres, unified, 10 Hz, velocities) frame.

    Event coordinates get the same metre conversion and direction unification."""
    mid = match.match_id
    for table in ("events", "positions", "frames", "players", "matches"):
        conn.execute(f"DELETE FROM {table} WHERE match_id = ?", (mid,))
    conn.execute("INSERT INTO matches VALUES (?, ?, ?, ?, ?)",
                 (mid, "metrica", cfg.target_fps, cfg.pitch_length, cfg.pitch_width))
    players = sorted({c[:-2] for c in trk.columns if c.endswith("_x") and c != "ball_x"})
    conn.executemany("INSERT INTO players VALUES (?, ?, ?)", [(mid, p, p[0]) for p in players])

    fr = trk[["frame", "period", "time", "ball_x", "ball_y", "ball_vx", "ball_vy"]].copy()
    fr.insert(0, "match_id", mid)
    conn.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     fr.astype(object).where(fr.notna(), None).itertuples(index=False, name=None))

    longs = []
    for p in players:
        part = trk[["frame", f"{p}_x", f"{p}_y", f"{p}_vx", f"{p}_vy"]].dropna(subset=[f"{p}_x"])
        part.columns = ["frame_id", "x", "y", "vx", "vy"]
        part.insert(1, "player_id", p)
        longs.append(part)
    pos = pd.concat(longs)
    pos.insert(0, "match_id", mid)
    conn.executemany("INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?)",
                     pos.astype(object).where(pos.notna(), None).itertuples(index=False, name=None))

    ev = match.events.copy()
    for c in ("start_x", "end_x"):
        ev[c] = (ev[c] - 0.5) * cfg.pitch_length
    for c in ("start_y", "end_y"):
        ev[c] = -(ev[c] - 0.5) * cfg.pitch_width
    flip = ev["period"].isin(flipped_periods(match.tracking, centre=0.5))  # same rotation as the tracking
    ev.loc[flip, ["start_x", "start_y", "end_x", "end_y"]] *= -1
    ev.insert(0, "match_id", mid)
    conn.executemany(
        "INSERT INTO events (match_id, team, type, subtype, period, start_frame, end_frame, from_player,"
        " to_player, start_x, start_y, end_x, end_y) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ev.astype(object).where(ev.notna(), None).itertuples(index=False, name=None))
    conn.commit()


def frames_around_event(conn: sqlite3.Connection, match_id: str, event_type: str,
                        before_s: float = 4.0, after_s: float = 2.0) -> pd.DataFrame:
    """All player positions from ``before_s`` before to ``after_s`` after each event of a type.

    Event frames are raw 25 Hz ids; stored frames are the 10 Hz subset, so the range join
    picks whichever stored frames fall inside the window.
    """
    q = """
    SELECT e.event_id, p.frame_id, p.player_id, p.x, p.y, p.vx, p.vy
    FROM events e
    JOIN positions p
      ON p.match_id = e.match_id
     AND p.frame_id BETWEEN e.start_frame - :b AND e.start_frame + :a
    WHERE e.match_id = :m AND e.type = :t
    ORDER BY e.event_id, p.frame_id, p.player_id
    """
    return pd.read_sql_query(q, conn, params={"m": match_id, "t": event_type,
                                              "b": int(before_s * 25), "a": int(after_s * 25)})
