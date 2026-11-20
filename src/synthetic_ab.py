"""Synthetic recordings in the schema of the repo frame table, with the same structure as the real data:
four attack recordings (bursts of injected frames separated by quiet stretches, then a long silent gap and
an attack-free tail that is the SAME frame sequence in every attack file) and one ambient recording.
For tests and dry runs only. NOT real data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CLASSES, FEATURES

SOURCES = {"dos": ("DoS", 1000.0), "fuzzy": ("Fuzzy", 900.0), "gear": ("Gear", 800.0), "rpm": ("RPM", 700.0)}


def _frames(ts, ids, dlc, data, label, source, flag):
    d = {"ts": np.asarray(ts, dtype=np.float64), "id": np.asarray(ids, dtype=np.int32),
         "dlc": np.asarray(dlc, dtype=np.int32)}
    for i in range(8):
        d[f"d{i}"] = data[:, i].astype(np.int32)
    d["label"] = label
    d["source"] = source
    d["flag"] = flag
    return pd.DataFrame(d)


def make_recordings(seed: int = 0, n_bursts: int = 12, burst_len: float = 2.0, quiet: float = 3.0,
                    tail_frames: int = 1500, gap: float = 40.0):
    rng = np.random.default_rng(seed)
    pool = rng.integers(0, 256, size=(80, 8))
    pool_ids = rng.integers(0x100, 0x7FF, size=80)
    pool_dlc = np.where(rng.random(80) < 0.15, 2, 8)
    pool[pool_dlc == 2, 2:] = 0
    w = 1.0 / np.arange(1, 81)
    w /= w.sum()

    def normal_frames(ts):
        k = rng.choice(80, size=len(ts), p=w)
        return pool_ids[k], pool_dlc[k], pool[k]

    tail_k = rng.choice(80, size=tail_frames, p=w)       # the shared tail: identical in every attack file
    parts, meta = [], {"bursts": n_bursts, "tail_frames": tail_frames, "tails": {}}
    for src, (label, t0) in SOURCES.items():
        ts_r, ts_t = [], []
        t = t0
        for _ in range(n_bursts):
            ts_r.append(np.arange(t, t + quiet, 0.02))
            t += quiet
            ts_r.append(np.arange(t, t + burst_len, 0.02))
            ts_t.append(np.arange(t + 0.001, t + burst_len, 0.002))
            t += burst_len
        ts_r = np.concatenate(ts_r)
        ts_t = np.concatenate(ts_t)
        ids, dlc, data = normal_frames(ts_r)
        parts.append(_frames(ts_r, ids, dlc, data, "Normal", src, "R"))
        n = len(ts_t)
        if src == "dos":
            tid, tdata = np.zeros(n), np.zeros((n, 8))
        elif src == "rpm":
            tid, tdata = np.full(n, 0x316), np.tile([0x45, 0x29, 0x24, 0xFF, 0x29, 0x24, 0x00, 0xFF], (n, 1))
        elif src == "gear":
            tid, tdata = np.full(n, 0x43F), np.tile([0x01, 0x45, 0x60, 0xFF, 0x6B, 0, 0, 0], (n, 1))
        else:
            tid, tdata = rng.integers(0, 0x800, size=n), rng.integers(0, 256, size=(n, 8))
        parts.append(_frames(ts_t, tid, np.full(n, 8), tdata, label, src, "T"))
        t_tail = t + gap
        ts_tail = t_tail + np.arange(tail_frames) * 0.02
        parts.append(_frames(ts_tail, pool_ids[tail_k], pool_dlc[tail_k], pool[tail_k], "Normal", src, "R"))
        meta["tails"][src] = tail_frames
    ts_n = np.arange(5000.0, 5050.0, 0.01)
    ids, dlc, data = normal_frames(ts_n)
    parts.append(_frames(ts_n, ids, dlc, data, "Normal", "normal", "R"))

    df = pd.concat(parts, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    df["sig_id"] = df.groupby(["id", "dlc"] + [f"d{i}" for i in range(8)], sort=False).ngroup().astype(np.int32)
    for c in FEATURES:
        df[c] = df[c].astype(np.int32)
    df["label"] = pd.Categorical(df["label"], categories=CLASSES)
    df["source"] = df["source"].astype("category")
    df["flag"] = df["flag"].astype("category")
    return df, meta
