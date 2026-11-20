"""Exact-signature lookup baseline (Master Context Pack, D20): predict the training label of the test
frame's signature; predict `fallback` (Normal) when the signature never occurs in training.
If a signature occurs under several labels the most frequent one wins (the real data has none)."""
from __future__ import annotations

import numpy as np

from .common import N_CLASSES


class LookupModel:
    def __init__(self, fallback: int = 0):
        self.fallback = fallback

    def fit(self, sig: np.ndarray, y: np.ndarray):
        n = int(sig.max()) + 1
        counts = np.bincount(sig.astype(np.int64) * N_CLASSES + y.astype(np.int64),
                             minlength=n * N_CLASSES).reshape(n, N_CLASSES)
        self.label_ = counts.argmax(axis=1).astype(np.int8)
        self.known_ = counts.sum(axis=1) > 0
        return self

    def predict(self, sig: np.ndarray) -> np.ndarray:
        sig = sig.astype(np.int64)
        inside = sig < len(self.label_)
        known = np.zeros(len(sig), dtype=bool)
        known[inside] = self.known_[sig[inside]]
        pred = np.full(len(sig), self.fallback, dtype=np.int8)
        pred[known] = self.label_[sig[known]]
        return pred
