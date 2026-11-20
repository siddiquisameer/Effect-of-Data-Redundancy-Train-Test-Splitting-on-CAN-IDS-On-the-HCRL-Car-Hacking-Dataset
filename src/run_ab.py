"""Technique A vs Technique B experiments -> results table in exactly the format of results_table_template.csv.

Real run (only after config.yaml has frozen: true AND validity.tolerance_fixed: true):
    python -m src.run_ab --table data/processed/frames.parquet --template results/results_table_template.csv

Dry run on synthetic recordings (safe any time, never touches real data):
    python -m src.run_ab --synthetic --models Lookup DT --seeds 42 43

Output files (never overwritten):
    results/ab_results_partial.csv   rows in the template format, appended after every finished run
    results/ab_run_log.csv           one line per run: sizes, seconds, config hash, notes
    results/ab_split_summary_<stamp>.csv, results/ab_cuts_<stamp>.csv   what each split looks like
    results/results_table_filled_<stamp>.csv   the template filled in (same rows, same order, same header)
A crashed run is simply started again with the same command: finished runs are skipped.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from .common import CLASSES, N_CLASSES, config_hash, load_config
from .data_ab import load_table, prepare
from .lookup import LookupModel
from .metrics import confusion_matrix
from .models import build_model
from .run_experiments import check_split
from .splits_ab import seen_mask, split_a, split_b, train_from_test

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_COLS = ["technique", "seed", "model", "label", "tp", "fp", "fn",
                 "tp_seen", "fn_seen", "tp_unseen", "fn_unseen"]
NUM_COLS = TEMPLATE_COLS[4:]
LOG_COLS = ["date", "technique", "seed", "model", "status", "n_train", "n_test", "fit_seconds",
            "predict_seconds", "copied_to_seeds", "config_hash", "note"]
ALL_MODELS = ["Lookup", "DT", "LR", "RF"]

# numbers measured by the team on the real sanitised data (Master Context Pack, 7.8 and 7.9)
KNOWN_B_TEST = {"Normal": 2344847, "DoS": 108623, "Fuzzy": 76121, "RPM": 135084, "Gear": 118490}
KNOWN_B_SEEN = {"Normal": 0.9145, "DoS": 1.0, "Fuzzy": 0.0, "RPM": 1.0, "Gear": 1.0}
KNOWN_A_TEST = {"Normal": 2332511, "DoS": 117504, "Fuzzy": 98369, "RPM": 130979, "Gear": 119450}
KNOWN_A_SEEN = {"Normal": 0.97101, "DoS": 1.0, "Fuzzy": 0.0, "RPM": 1.0, "Gear": 1.0}
KNOWN_B_CUTS = {"dos": (1864.503, 3.004), "fuzzy": (1972.794, 3.013), "rpm": (1566.145, 3.005),
                "gear": (1562.469, 3.004), "normal": (405.149, None)}


# ------------------------------------------------------------------ template and rows
def make_template(seeds, models, techniques=("A", "B")):
    return [{"technique": t, "seed": str(s), "model": m, "label": lab, **{c: "" for c in NUM_COLS}}
            for t in techniques for s in seeds for m in models for lab in CLASSES]


def read_template(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != TEMPLATE_COLS:
            raise SystemExit(f"{path}: columns differ from the agreed template {TEMPLATE_COLS}")
        return list(rd)


def counts_from(y_te, pred, seen) -> dict:
    cm = confusion_matrix(y_te, pred)
    cs = confusion_matrix(y_te[seen], pred[seen])
    cu = confusion_matrix(y_te[~seen], pred[~seen])
    tp = np.diag(cm)
    out = {}
    for i, lab in enumerate(CLASSES):
        out[lab] = {"tp": int(tp[i]), "fp": int(cm[:, i].sum() - tp[i]), "fn": int(cm[i].sum() - tp[i]),
                    "tp_seen": int(cs[i, i]), "fn_seen": int(cs[i].sum() - cs[i, i]),
                    "tp_unseen": int(cu[i, i]), "fn_unseen": int(cu[i].sum() - cu[i, i])}
    return out


def _append(path: Path, fields, rows):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def _done_keys(partial: Path) -> set:
    if not partial.exists() or partial.stat().st_size == 0:
        return set()
    n = {}
    with open(partial, newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != TEMPLATE_COLS:
            raise SystemExit(f"{partial} has different columns; use a new --out-dir")
        for r in rd:
            k = (r["technique"], int(r["seed"]), r["model"])
            n[k] = n.get(k, 0) + 1
    return {k for k, v in n.items() if v >= N_CLASSES}


def check_table(rows) -> list:
    """Invariants a correct table must satisfy. Returns a list of problems (empty = fine)."""
    problems, groups = [], {}
    for r in rows:
        if any(r[c] == "" for c in NUM_COLS):
            continue
        g = groups.setdefault((r["technique"], int(r["seed"]), r["model"]), {})
        g[r["label"]] = {c: int(r[c]) for c in NUM_COLS}
    support = {}
    for (t, s, m), g in groups.items():
        if set(g) != set(CLASSES):
            problems.append(f"{t} {s} {m}: not all five labels filled")
            continue
        if sum(v["fp"] for v in g.values()) != sum(v["fn"] for v in g.values()):
            problems.append(f"{t} {s} {m}: sum of fp differs from sum of fn")
        for lab, v in g.items():
            if v["tp"] != v["tp_seen"] + v["tp_unseen"]:
                problems.append(f"{t} {s} {m} {lab}: tp != tp_seen + tp_unseen")
            if v["fn"] != v["fn_seen"] + v["fn_unseen"]:
                problems.append(f"{t} {s} {m} {lab}: fn != fn_seen + fn_unseen")
            sup = support.setdefault((t, s, lab), v["tp"] + v["fn"])
            if sup != v["tp"] + v["fn"]:
                problems.append(f"{t} {s} {m} {lab}: test frames of this class differ between models")
    return problems


def fill_template(template_rows, partial: Path):
    have = {}
    with open(partial, newline="") as f:
        for r in csv.DictReader(f):
            have[(r["technique"], r["seed"], r["model"], r["label"])] = r
    keys = {(r["technique"], r["seed"], r["model"], r["label"]) for r in template_rows}
    extra = set(have) - keys
    if extra:
        raise SystemExit(f"{len(extra)} result rows are not in the template, e.g. {sorted(extra)[:3]}")
    filled = []
    for r in template_rows:
        k = (r["technique"], r["seed"], r["model"], r["label"])
        filled.append({**r, **({c: have[k][c] for c in NUM_COLS} if k in have else {})})
    return filled


# ------------------------------------------------------------------ experiment
def _stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_ab(arr, cfg, ab, out_dir, techniques, seeds, models, log=print) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial, logf = out_dir / "ab_results_partial.csv", out_dir / "ab_run_log.csv"
    names = ab["model_names"]
    chash = config_hash(cfg)
    done = _done_keys(partial)
    X, y, sig, n = arr["X"], arr["y"], arr["sig"], len(arr["y"])
    fb = CLASSES.index(ab["lookup_fallback"])
    det = set(ab["deterministic_for_B"])
    stamp = _stamp()
    summary, cuts_info, runs = [], {}, 0

    def todo(tech, seed):
        out = []
        for m in models:
            if tech == "B" and m in det:
                if (tech, seeds[0], names[m]) not in done and seed == seeds[0]:
                    out.append(m)
            elif (tech, seed, names[m]) not in done:
                out.append(m)
        return out

    test_b = None
    for tech in techniques:
        for seed in seeds:
            models_now = todo(tech, seed)
            if not models_now:
                log(f"[skip] {tech} seed {seed}: nothing left to run")
                continue
            if tech == "A":
                test_idx = split_a(y, seed, ab["test_fraction"])
            else:
                if test_b is None:
                    test_b, cuts_info = split_b(arr["ts"], arr["is_t"], arr["source"], ab["burst_gap_s"], ab["cut_fraction"])
                test_idx = test_b
            train_idx = train_from_test(n, test_idx)
            chk = check_split(y, test_idx, cfg["validity"])
            chk.pop("test_mask")
            seen = seen_mask(sig, train_idx, test_idx)
            y_te, y_tr = y[test_idx], y[train_idx]
            summary.extend(_summary_rows(tech, seed, y, train_idx, test_idx, seen))
            if not chk["valid"] and cfg["validity"]["on_invalid"] == "skip":
                note = f"invalid split: missing in train {chk['missing_train']}, missing in test {chk['missing_test']}, " \
                       f"outside tolerance {chk['out_of_tolerance']}"
                _append(logf, LOG_COLS, [{"date": datetime.now().isoformat(timespec="seconds"), "technique": tech,
                                          "seed": seed, "model": names[m], "status": "skipped_invalid", "note": note,
                                          "config_hash": chash} for m in models_now])
                log(f"[invalid] {tech} seed {seed}: {note}")
                continue
            Xtr = Xte = None
            for m in models_now:
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        t0 = time.perf_counter()
                        if m == "Lookup":
                            model = LookupModel(fb).fit(sig[train_idx], y_tr)
                        else:
                            if Xtr is None:
                                Xtr, Xte = X[train_idx], X[test_idx]
                            model = build_model(m, seed, cfg).fit(Xtr, y_tr)
                        fit_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    pred = model.predict(sig[test_idx]) if m == "Lookup" else model.predict(Xte)
                    pred_s = time.perf_counter() - t0
                    cnt = counts_from(y_te, pred, seen)
                    copy_to = [seed]
                    if tech == "B" and m in det:
                        copy_to = list(seeds)
                    rows = [{"technique": tech, "seed": s, "model": names[m], "label": lab, **cnt[lab]}
                            for s in copy_to for lab in CLASSES]
                    _append(partial, TEMPLATE_COLS, rows)
                    _append(logf, LOG_COLS, [{
                        "date": datetime.now().isoformat(timespec="seconds"), "technique": tech, "seed": seed,
                        "model": names[m], "status": "ok", "n_train": len(train_idx), "n_test": len(test_idx),
                        "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 2),
                        "copied_to_seeds": ";".join(map(str, copy_to)) if len(copy_to) > 1 else "",
                        "config_hash": chash, "note": "; ".join(sorted({x.category.__name__ for x in w}))}])
                    tp = sum(v["tp"] for v in cnt.values())
                    log(f"[ok] {tech} seed {seed} {names[m]}: accuracy {tp / len(test_idx):.4f}, fit {fit_s:.0f}s")
                    runs += 1
                    del model
                except Exception as e:                                  # keep going; retried next start
                    _append(logf, LOG_COLS, [{"date": datetime.now().isoformat(timespec="seconds"), "technique": tech,
                                              "seed": seed, "model": names[m], "status": "error",
                                              "note": f"{type(e).__name__}: {e}"[:300], "config_hash": chash}])
                    log(f"[error] {tech} seed {seed} {names[m]}: {type(e).__name__}: {e}")
            del Xtr, Xte
    if summary:
        _append(out_dir / f"ab_split_summary_{stamp}.csv", list(summary[0].keys()), summary)
    if cuts_info:
        rows = [{"source": arr["source_names"][s], **{k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()}}
                for s, c in cuts_info.items()]
        _append(out_dir / f"ab_cuts_{stamp}.csv", list(rows[0].keys()), rows)
    return runs, summary, cuts_info


def _summary_rows(tech, seed, y, train_idx, test_idx, seen):
    y_tr, y_te = y[train_idx], y[test_idx]
    rows = []
    for c, lab in enumerate(CLASSES):
        nt, ne = int((y_tr == c).sum()), int((y_te == c).sum())
        rows.append({"technique": tech, "seed": seed, "label": lab, "n_train": nt, "n_test": ne,
                     "test_share": round(ne / (nt + ne), 4) if nt + ne else "",
                     "seen_share": round(float(seen[y_te == c].mean()), 5) if ne else ""})
    return rows


def splits_only(arr, ab, techniques, seeds, out_dir, log=print):
    """Build the splits and describe them. Trains NO model, so it is safe before the settings are frozen."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y, sig, n = arr["y"], arr["sig"], len(arr["y"])
    summary, cuts = [], {}
    for tech in techniques:
        for seed in (seeds if tech == "A" else seeds[:1]):
            if tech == "A":
                test_idx = split_a(y, seed, ab["test_fraction"])
            else:
                test_idx, cuts = split_b(arr["ts"], arr["is_t"], arr["source"], ab["burst_gap_s"], ab["cut_fraction"])
            train_idx = train_from_test(n, test_idx)
            seen = seen_mask(sig, train_idx, test_idx)
            summary.extend(_summary_rows(tech, seed, y, train_idx, test_idx, seen))
            log(f"split {tech} seed {seed}: {len(train_idx):,} train, {len(test_idx):,} test")
    stamp = _stamp()
    if summary:
        _append(out_dir / f"ab_split_summary_{stamp}.csv", list(summary[0].keys()), summary)
    if cuts:
        rows = [{"source": arr["source_names"][s], **{k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()}}
                for s, c in cuts.items()]
        _append(out_dir / f"ab_cuts_{stamp}.csv", list(rows[0].keys()), rows)
    return summary, cuts


def finalize(template_rows, out_dir, log=print):
    out_dir = Path(out_dir)
    partial = out_dir / "ab_results_partial.csv"
    if not partial.exists():
        log("nothing to finalize yet")
        return None
    filled = fill_template(template_rows, partial)
    problems = check_table(filled)
    path = out_dir / f"results_table_filled_{_stamp()}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEMPLATE_COLS)
        w.writeheader()
        w.writerows(filled)
    blank = sum(1 for r in filled if r["tp"] == "")
    log(f"filled table: {path}  ({len(filled) - blank} of {len(filled)} rows filled)")
    for p in problems:
        log("  CHECK FAILED: " + p)
    if not problems:
        log("  all self-checks passed (support equal across models, sum fp = sum fn, seen + unseen = total)")
    return path, problems


def compare_known(summary, cuts, source_names, log=print):
    """Compare our splits with the numbers the team measured independently (only meaningful on the full real data)."""
    def get(t, lab, key, agg=np.mean):
        v = [float(r[key]) for r in summary if r["technique"] == t and r["label"] == lab and r[key] != ""]
        return agg(v) if v else None
    ok = True
    for lab in CLASSES:
        b = get("B", lab, "n_test")
        a = get("A", lab, "n_test")
        for tag, got, want, tol in [("B test frames", b, KNOWN_B_TEST[lab], 0), ("A test frames", a, KNOWN_A_TEST[lab], 0),
                                    ("B seen share", get("B", lab, "seen_share"), KNOWN_B_SEEN[lab], 2e-4),
                                    ("A seen share", get("A", lab, "seen_share"), KNOWN_A_SEEN[lab], 3e-4)]:
            if got is None:
                continue
            good = abs(got - want) <= tol
            ok &= good
            log(f"  {'PASS' if good else 'DIFF'}  {lab:7s} {tag}: ours {got:.5g}, team {want:.5g}")
    for s, c in (cuts or {}).items():
        nm = source_names[s]
        if nm in KNOWN_B_CUTS:
            rel, q = KNOWN_B_CUTS[nm]
            good = abs(c["cut_rel"] - rel) < 0.01 and (q is None or abs(c["quiet_len"] - q) < 0.01)
            ok &= good
            log(f"  {'PASS' if good else 'DIFF'}  cut in {nm}: ours {c['cut_rel']:.3f} s (quiet {c['quiet_len']:.3f}), "
                f"team {rel} s, bursts {c['n_bursts']}")
    return ok


def guard(cfg) -> list:
    problems = []
    if not cfg.get("frozen"):
        problems.append("config.yaml: frozen is false (model settings must be written down first)")
    if not cfg["validity"].get("tolerance_fixed"):
        problems.append("config.yaml: validity.tolerance_fixed is false (the supervisor must confirm the validity tolerance)")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--ab-config", default=str(ROOT / "config_ab.yaml"))
    ap.add_argument("--table", default=None, help="frames.parquet of the repo pipeline (real run)")
    ap.add_argument("--template", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--techniques", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true", help="dry run on synthetic recordings")
    ap.add_argument("--skip-sanitise", action="store_true", help="only for the small repo sample: its gaps are not real")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th row after sanitising (pilot)")
    ap.add_argument("--known-checks", action="store_true", help="compare with the team's measured numbers (full data)")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--splits-only", action="store_true", help="build and describe the splits, train no model")
    a = ap.parse_args(argv)
    cfg, ab = load_config(a.config), load_config(a.ab_config)
    seeds = a.seeds or cfg["seeds"]
    template_path = Path(a.template or ROOT / ab["template"])

    if a.synthetic:
        from .synthetic_ab import make_recordings
        out_dir = Path(a.out_dir or ROOT / "results" / "ab_dry_run")
        df, _ = make_recordings(seed=0)
        arr = prepare(df, True, ab["tail_gap_s"], a.stride)
        template = make_template(cfg["seeds"], [ab["model_names"][m] for m in ALL_MODELS])
        print(f"DRY RUN on synthetic recordings ({len(arr['y']):,} frames after sanitising) -> {out_dir}")
    else:
        problems = [] if a.splits_only else guard(cfg)
        if problems:
            print("Refusing to run on real data:\n  - " + "\n  - ".join(problems), file=sys.stderr)
            return 2
        if not a.table:
            print("give --table path/to/frames.parquet", file=sys.stderr)
            return 2
        out_dir = Path(a.out_dir or ROOT / "results")
        template = [] if a.splits_only else read_template(template_path)
        df = load_table(a.table)
        arr = prepare(df, not a.skip_sanitise, ab["tail_gap_s"], a.stride)
        del df
        print(f"loaded {arr['n_input']:,} frames, {len(arr['y']):,} after sanitising; removed per recording: {arr['removed']}")
        print("class counts:", dict(zip(CLASSES, np.bincount(arr["y"], minlength=N_CLASSES).tolist())))
    if a.splits_only:
        summary, cuts = splits_only(arr, ab, a.techniques, seeds, out_dir)
        if a.known_checks:
            print("comparison with the numbers measured by the team:")
            compare_known(summary, cuts, arr["source_names"])
        return 0
    if a.finalize_only:
        finalize(template, out_dir)
        return 0
    runs, summary, cuts = run_ab(arr, cfg, ab, out_dir, a.techniques, seeds, a.models)
    if a.known_checks:
        print("comparison with the numbers measured by the team:")
        compare_known(summary, cuts, arr["source_names"])
    finalize(template, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
