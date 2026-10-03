"""Loaders for the Metrica Sports open sample data.

Every loader returns a ``Match`` whose tracking frame is wide:
``period, frame, time, <player>_x, <player>_y, ..., ball_x, ball_y``
with coordinates still normalized to [0, 1] (origin top-left), exactly as provided.
Player columns are prefixed ``H_`` (home) or ``A_`` (away).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Match:
    match_id: str
    tracking: pd.DataFrame
    events: pd.DataFrame
    fps: int = 25

    @property
    def players(self) -> list[str]:
        return sorted({c[:-2] for c in self.tracking.columns if c.endswith("_x") and c != "ball_x"})


EVENT_COLS = ["team", "type", "subtype", "period", "start_frame", "end_frame",
              "from_player", "to_player", "start_x", "start_y", "end_x", "end_y"]


def _read_csv_team(path: Path, prefix: str) -> pd.DataFrame:
    names = pd.read_csv(path, header=None, skiprows=2, nrows=1).iloc[0].tolist()
    cols = ["period", "frame", "time"]
    for i in range(3, len(names), 2):
        cols += [f"{names[i]}_x", f"{names[i]}_y"]
    df = pd.read_csv(path, header=None, skiprows=3, low_memory=False)
    df.columns = cols[: df.shape[1]]
    df = df.rename(columns={"Ball_x": "ball_x", "Ball_y": "ball_y"})
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c.startswith("Player")})


def load_csv_game(root: Path | str, game: int) -> Match:
    """Sample_Game_1 / Sample_Game_2 (Metrica CSV format)."""
    d = Path(root) / f"Sample_Game_{game}"
    home = _read_csv_team(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", "H")
    away = _read_csv_team(d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", "A")
    away = away.drop(columns=["period", "frame", "time", "ball_x", "ball_y"])
    trk = pd.concat([home.drop(columns=["ball_x", "ball_y"]), away, home[["ball_x", "ball_y"]]], axis=1)

    ev = pd.read_csv(d / f"Sample_Game_{game}_RawEventsData.csv")
    ev.columns = EVENT_COLS[:2] + ["subtype", "period", "start_frame", "start_time",
                                   "end_frame", "end_time"] + EVENT_COLS[6:]
    ev["team"] = ev["team"].map({"Home": "H", "Away": "A"})
    for c in ("from_player", "to_player"):
        ev[c] = ev[c].where(ev[c].isna(), ev["team"] + "_" + ev[c].astype(str))
    return Match(f"metrica_{game}", trk, ev[EVENT_COLS])


def load_epts_game(root: Path | str, game: int = 3) -> Match:
    """Sample_Game_3 (FIFA EPTS tracking + JSON events).

    The tracking line holds 22 player slots, but which player sits in which slot
    changes per ``DataFormatSpecification`` frame range (substitutions), so the
    slot order is resolved from the metadata for every frame.
    """
    d = Path(root) / f"Sample_Game_{game}"
    meta = ET.parse(d / f"Sample_Game_{game}_metadata.xml").getroot()
    teams = [t.get("id") for t in meta.iter("Team")]
    side = {p.get("id"): ("H" if p.get("teamId") == teams[0] else "A") for p in meta.iter("Player")}
    channel_player = {pc.get("id"): pc.get("playerId") for pc in meta.iter("PlayerChannel")}
    specs = []
    for sp in meta.iter("DataFormatSpecification"):
        order = [channel_player[r.get("playerChannelId")] for r in sp.iter("PlayerChannelRef")
                 if r.get("playerChannelId").endswith("_x")]
        specs.append((int(sp.get("startFrame")), int(sp.get("endFrame")), order))
    params = {p.findtext("Name"): p.findtext("Value") for p in meta.iter("ProviderParameter")}
    second_half = int(params["second_half_start"])

    pids = sorted({p for _, _, o in specs for p in o})
    col = {p: i for i, p in enumerate(pids)}
    frames, xy, ball = [], [], []
    with open(d / f"Sample_Game_{game}_tracking.txt") as f:
        for line in f:
            fr, players, b = line.strip().split(":")
            fr = int(fr)
            order = next(o for s0, s1, o in specs if s0 <= fr <= s1)
            row = np.full(2 * len(pids), np.nan)
            for p, pos in zip(order, players.split(";")):
                x, y = pos.split(",")
                row[2 * col[p]], row[2 * col[p] + 1] = float(x), float(y)
            bx, by = b.split(",")[:2]
            frames.append(fr); xy.append(row); ball.append((float(bx), float(by)))

    names = [f"{side[p]}_{p}" for p in pids]
    trk = pd.DataFrame(np.asarray(xy), columns=[f"{n}_{a}" for n in names for a in "xy"])
    trk[["ball_x", "ball_y"]] = np.asarray(ball)
    trk.insert(0, "frame", frames)
    trk.insert(0, "period", np.where(trk["frame"] >= second_half, 2, 1))
    trk.insert(2, "time", trk["frame"] / 25)

    raw = json.load(open(d / f"Sample_Game_{game}_events.json"))["data"]
    rows = []
    for e in raw:
        t = "H" if e["team"]["id"] == teams[0] else "A"
        sub = e["subtypes"]
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        frm, to = e.get("from"), e.get("to")
        rows.append({
            "team": t, "type": e["type"]["name"], "subtype": sub, "period": e["period"],
            "start_frame": e["start"]["frame"], "end_frame": e["end"]["frame"],
            "from_player": f"{t}_{frm['id']}" if frm else None,
            "to_player": f"{t}_{to['id']}" if to else None,
            "start_x": e["start"]["x"], "start_y": e["start"]["y"],
            "end_x": e["end"]["x"], "end_y": e["end"]["y"],
        })
    return Match(f"metrica_{game}", trk, pd.DataFrame(rows, columns=EVENT_COLS))


def load_all(root: Path | str = "data/metrica/data") -> list[Match]:
    return [load_csv_game(root, 1), load_csv_game(root, 2), load_epts_game(root, 3)]
