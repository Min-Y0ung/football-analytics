"""Load the 3 Metrica matches, preprocess, and write them to SQLite.

    python scripts/build_db.py --data data/metrica/data --db data/football.db
"""
import argparse
import time

from stfootball.db import connect, load_match
from stfootball.io.metrica import load_all
from stfootball.preprocess import preprocess

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/metrica/data")
ap.add_argument("--db", default="data/football.db")
args = ap.parse_args()

conn = connect(args.db)
for m in load_all(args.data):
    t0 = time.time()
    trk = preprocess(m)
    load_match(conn, m, trk)
    print(f"{m.match_id}: {len(trk):,} frames @10Hz, {len(m.events):,} events ({time.time() - t0:.0f}s)")
for table in ("matches", "players", "frames", "positions", "events"):
    print(f"{table:10s} {conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>10,}")
