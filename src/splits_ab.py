"""Technique A and Technique B splits, exactly as defined in the Master Context Pack (6.6, 6.7, 6.9).

A: stratified random 80/20. For every class exactly round(0.2 * N_c) frames go to the test side.
B: chronological block split inside each recording (timestamps are comparable only inside one file).
   Bursts = runs of T frames; a gap above `burst_gap` seconds between T frames starts a new burst.
   tau = t0 + 0.8 * (t_end - t0). If tau falls inside a burst it moves to the quiet stretch after that
   burst; otherwise the quiet stretch that contains it is used. The cut is the midpoint of that quiet
   stretch. Train = t <= cut, test = t > cut. No frame is discarded. A recording without T frames is cut
   at 80% of its duration. B is deterministic (the seed does not change it).
seen/unseen: a test frame is seen if its exact signature occurs in the training side.
"""
from __future__ import annotations

import numpy as np

from .common import N_CLASSES


def split_a(y: np.ndarray, seed: int, test_fraction: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        k = int(round(test_fraction * len(idx)))
        if k > 0:
            parts.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(parts)).astype(np.int64)


def burst_cut(ts: np.ndarray, is_t: np.ndarray, burst_gap: float = 1.0, frac: float = 0.8) -> dict:
    t0, t1 = float(ts[0]), float(ts[-1])
    tau = t0 + frac * (t1 - t0)
    if not is_t.any():
        return {"cut": tau, "cut_rel": tau - t0, "n_bursts": 0, "quiet_len": float("nan"), "t0": t0, "t1": t1}
    tt = ts[is_t]
    new = np.flatnonzero(np.diff(tt) > burst_gap) + 1
    starts = np.concatenate([[tt[0]], tt[new]])
    ends = np.concatenate([tt[new - 1], [tt[-1]]])
    k = int(np.searchsorted(starts, tau, side="right")) - 1       # last burst that starts at or before tau
    if k >= 0:                                                    # inside burst k or in the quiet after it
        lo = float(ends[k])
        hi = float(starts[k + 1]) if k + 1 < len(starts) else t1
    else:                                                         # before the first burst
        lo, hi = t0, float(starts[0])
    cut = 0.5 * (lo + hi)
    return {"cut": cut, "cut_rel": cut - t0, "n_bursts": int(len(starts)), "quiet_len": hi - lo, "t0": t0, "t1": t1}


def split_b(ts: np.ndarray, is_t: np.ndarray, source: np.ndarray, burst_gap: float = 1.0, frac: float = 0.8):
    """-> sorted test indices and {source code: cut info}."""
    test = np.zeros(len(ts), dtype=bool)
    info = {}
    for s in np.unique(source):
        idx = np.flatnonzero(source == s)
        t = ts[idx]
        if np.any(np.diff(t) < 0):
            raise ValueError("timestamps go backwards inside one recording; the table must be in time order")
        c = burst_cut(t, is_t[idx], burst_gap, frac)
        test[idx[t > c["cut"]]] = True
        info[int(s)] = c
    return np.flatnonzero(test).astype(np.int64), info


def train_from_test(n: int, test_idx: np.ndarray) -> np.ndarray:
    m = np.ones(n, dtype=bool)
    m[test_idx] = False
    return np.flatnonzero(m)


def seen_mask(sig: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    """True for test frames whose exact signature occurs in the training side."""
    present = np.zeros(int(sig.max()) + 1, dtype=bool)
    present[sig[train_idx]] = True
    return present[sig[test_idx]]
