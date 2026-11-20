"""Loading and sanitising the frame table of the repo pipeline (src/build_dataset.py).

Table columns: ts, id, dlc, d0..d7, label, source, flag, sig_id (data64 is not needed here).
Sanitisation (Master Context Pack, 6.5): in every recording find the largest gap between consecutive
timestamps; if it exceeds `tail_gap` seconds, drop every frame after it (the attack-free tail that is a
copy of the same recording in all four attack files).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CLASSES, FEATURES, labels_to_int

TABLE_COLUMNS = ["ts"] + FEATURES + ["label", "source", "flag", "sig_id"]


def load_table(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=TABLE_COLUMNS)
    return pd.read_csv(path, usecols=TABLE_COLUMNS)


def _label_codes(lab: pd.Series) -> np.ndarray:
    if hasattr(lab, "cat") and list(lab.cat.categories) == CLASSES:      # fast path, no strings
        return lab.cat.codes.to_numpy().astype(np.int8)
    return labels_to_int(lab)


def tail_mask(ts: np.ndarray, src: np.ndarray, tail_gap: float = 1.0):
    """-> keep mask and {source code: frames removed}. Frames of one source must be in time order."""
    keep = np.ones(len(ts), dtype=bool)
    removed = {}
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        t = ts[idx]
        if len(t) < 2:
            continue
        gaps = np.diff(t)
        j = int(np.argmax(gaps))
        if gaps[j] > tail_gap:
            keep[idx[j + 1:]] = False
            removed[int(s)] = int(len(idx) - (j + 1))
    return keep, removed


def prepare(df: pd.DataFrame, sanitise: bool = True, tail_gap: float = 1.0, stride: int = 1) -> dict:
    """Table -> numpy arrays. Order of operations: sanitise first, then keep every `stride`-th row."""
    ts = df["ts"].to_numpy(np.float64)
    src_cat = df["source"].astype("category")
    src = src_cat.cat.codes.to_numpy().astype(np.int8)
    names = [str(c) for c in src_cat.cat.categories]
    keep, removed = tail_mask(ts, src, tail_gap) if sanitise else (np.ones(len(ts), bool), {})
    sel = np.flatnonzero(keep)
    if stride > 1:
        sel = sel[::stride]
    return {
        "X": df[FEATURES].to_numpy(np.float32)[sel],
        "y": _label_codes(df["label"])[sel],
        "sig": df["sig_id"].to_numpy(np.int32)[sel],
        "ts": ts[sel],
        "source": src[sel],
        "source_names": names,
        "is_t": (df["flag"] == "T").to_numpy()[sel],
        "removed": {names[k]: v for k, v in removed.items()},
        "n_input": int(len(df)),
    }
