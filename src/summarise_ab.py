"""Summary of a filled results table: mean and standard deviation over seeds (descriptive, Context Pack 6.4, 6.11).

    python -m src.summarise_ab results/results_table_filled_<stamp>.csv

Per technique and model: macro-F1, accuracy (with the all-Normal baseline of the same test set), per-class recall,
recall on seen and on unseen test frames. Everything is computed from the counts tp, fp, fn, tp_seen, fn_seen,
tp_unseen, fn_unseen. A class with no test frames has recall NaN (not 0)."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CLASSES

NUM = ["tp", "fp", "fn", "tp_seen", "fn_seen", "tp_unseen", "fn_unseen"]


def _div(a, b):
    return a / b if b > 0 else np.nan


def per_run(g: pd.DataFrame) -> dict:
    g = g.set_index("label").reindex(CLASSES)
    tp, fp, fn = g["tp"].to_numpy(float), g["fp"].to_numpy(float), g["fn"].to_numpy(float)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    n = (tp + fn).sum()
    out = {"macro_f1": f1.mean(), "accuracy": tp.sum() / n,
           "baseline_macro_f1": 2 * ((tp[0] + fn[0]) / n) / (5 * (1 + (tp[0] + fn[0]) / n))}
    for i, lab in enumerate(CLASSES):
        out[f"recall_{lab}"] = _div(tp[i], tp[i] + fn[i])
        out[f"recall_seen_{lab}"] = _div(g["tp_seen"].iloc[i], g["tp_seen"].iloc[i] + g["fn_seen"].iloc[i])
        out[f"recall_unseen_{lab}"] = _div(g["tp_unseen"].iloc[i], g["tp_unseen"].iloc[i] + g["fn_unseen"].iloc[i])
        out[f"seen_share_{lab}"] = _div(g["tp_seen"].iloc[i] + g["fn_seen"].iloc[i], tp[i] + fn[i])
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["tp"].notna() & (df["tp"].astype(str) != "")].copy()
    for c in NUM:
        df[c] = df[c].astype(int)
    runs = [{"technique": t, "model": m, "seed": s, **per_run(g)}
            for (t, m, s), g in df.groupby(["technique", "model", "seed"], sort=False)]
    r = pd.DataFrame(runs)
    metrics = [c for c in r.columns if c not in ("technique", "model", "seed")]
    rows = []
    for (t, m), g in r.groupby(["technique", "model"], sort=False):
        for k in metrics:
            rows.append({"technique": t, "model": m, "metric": k, "mean": g[k].mean(), "sd": g[k].std(ddof=1) if len(g) > 1 else 0.0,
                         "n_seeds": len(g)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python -m src.summarise_ab results/results_table_filled_<stamp>.csv", file=sys.stderr)
        return 2
    src = Path(argv[0])
    out = src.parent / f"summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
    s = summarise(pd.read_csv(src, dtype=str, keep_default_na=False).replace("", np.nan).dropna(subset=["tp"]))
    s.to_csv(out, index=False)
    print(f"written {out}")
    key = s[s["metric"].isin(["macro_f1", "accuracy", "baseline_macro_f1"])].pivot_table(
        index=["technique", "model"], columns="metric", values="mean", sort=False)
    print(key.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
