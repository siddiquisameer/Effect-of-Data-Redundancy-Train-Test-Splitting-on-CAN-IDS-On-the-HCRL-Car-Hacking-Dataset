import copy
import csv
import tempfile
from pathlib import Path

import yaml

from src.common import load_config
from src.data_ab import prepare
from src.run_ab import (NUM_COLS, TEMPLATE_COLS, check_table, counts_from, fill_template, finalize, main,
                        make_template, run_ab)
from src.synthetic_ab import make_recordings

CFG = load_config("config.yaml")
CFG["models"]["RF"].update(n_estimators=5)
CFG["validity"].update(min_test_share=0.0, max_test_share=1.0)       # synthetic shares are not the point here
AB = load_config("config_ab.yaml")
DF, _ = make_recordings(seed=2, n_bursts=8)
ARR = prepare(DF, True, 1.0)
SEEDS = [42, 43]


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_full_dry_run_fills_template_and_passes_self_checks():
    out = Path(tempfile.mkdtemp())
    run_ab(ARR, CFG, AB, out, ["A", "B"], SEEDS, ["Lookup", "DT", "LR", "RF"], log=lambda *_: None)
    template = make_template(SEEDS, [AB["model_names"][m] for m in ["Lookup", "DT", "LR", "RF"]])
    path, problems = finalize(template, out, log=lambda *_: None)
    rows = _rows(path)
    assert list(rows[0].keys()) == TEMPLATE_COLS and len(rows) == len(template) == 80
    assert all(r[c] != "" for r in rows for c in NUM_COLS)
    assert [(r["technique"], r["seed"], r["model"], r["label"]) for r in rows] == \
           [(r["technique"], str(r["seed"]), r["model"], r["label"]) for r in template]      # same order as the template
    assert problems == []
    b = {(r["seed"], r["model"], r["label"]): r for r in rows if r["technique"] == "B"}
    for lab in ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]:                   # deterministic models are copied across seeds
        for m in ["Lookup", "LogisticRegression"]:
            assert all(b[("42", m, lab)][c] == b[("43", m, lab)][c] for c in NUM_COLS)


def test_resume_runs_nothing_twice():
    out = Path(tempfile.mkdtemp())
    first = run_ab(ARR, CFG, AB, out, ["A", "B"], SEEDS, ["Lookup", "DT"], log=lambda *_: None)[0]
    n_rows = len(_rows(out / "ab_results_partial.csv"))
    second = run_ab(ARR, CFG, AB, out, ["A", "B"], SEEDS, ["Lookup", "DT"], log=lambda *_: None)[0]
    assert first == 2 + 2 + 1 + 2 and second == 0          # A: 2 seeds x 2 models; B: Lookup once, DT twice
    assert len(_rows(out / "ab_results_partial.csv")) == n_rows


def test_lookup_is_perfect_on_seen_frames_and_normal_on_unseen():
    out = Path(tempfile.mkdtemp())
    run_ab(ARR, CFG, AB, out, ["A"], [42], ["Lookup"], log=lambda *_: None)
    for r in _rows(out / "ab_results_partial.csv"):
        assert int(r["fn_seen"]) == 0                              # a seen signature has a single label here
        if r["label"] == "Normal":
            assert int(r["fn_unseen"]) == 0                        # unseen frames are predicted Normal
        else:
            assert int(r["tp_unseen"]) == 0


def test_check_table_catches_wrong_numbers():
    out = Path(tempfile.mkdtemp())
    run_ab(ARR, CFG, AB, out, ["A"], [42], ["DT"], log=lambda *_: None)
    template = make_template([42], ["DecisionTree"], techniques=("A",))
    filled = fill_template(template, out / "ab_results_partial.csv")
    assert check_table(filled) == []
    bad = copy.deepcopy(filled)
    bad[0]["tp_seen"] = str(int(bad[0]["tp_seen"]) + 1)
    assert check_table(bad) != []


def test_row_not_in_template_is_rejected():
    out = Path(tempfile.mkdtemp())
    run_ab(ARR, CFG, AB, out, ["A"], [42], ["DT"], log=lambda *_: None)
    try:
        fill_template(make_template([43], ["DecisionTree"], techniques=("A",)), out / "ab_results_partial.csv")
    except SystemExit:
        return
    raise AssertionError("rows outside the template must be rejected")


def test_counts_from_small_example():
    import numpy as np
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 4]); p = np.array([0, 0, 0, 1, 1, 1, 2, 0, 3, 4])
    seen = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    c = counts_from(y, p, seen)
    assert c["Normal"] == {"tp": 3, "fp": 1, "fn": 1, "tp_seen": 2, "fn_seen": 0, "tp_unseen": 1, "fn_unseen": 1}
    assert c["DoS"]["fp"] == 1 and c["Fuzzy"]["fn"] == 1 and c["Fuzzy"]["fn_unseen"] == 1


def test_real_run_refused_until_flags_are_true():
    cfg = load_config("config.yaml")
    cfg["frozen"] = False
    cfg["validity"]["tolerance_fixed"] = False
    tmp = Path(tempfile.mkdtemp()) / "c.yaml"
    tmp.write_text(yaml.safe_dump(cfg))
    assert main(["--config", str(tmp), "--table", "none.parquet"]) == 2


def test_summary_matches_metrics_module():
    import numpy as np
    import pandas as pd
    from src.metrics import evaluate
    from src.summarise_ab import per_run
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 4]); p = np.array([0, 0, 0, 1, 1, 1, 2, 0, 3, 4])
    seen = np.ones(10, dtype=bool)
    c = counts_from(y, p, seen)
    g = pd.DataFrame([{"label": lab, **v} for lab, v in c.items()])
    r = per_run(g)
    ref = evaluate(y, p)
    assert abs(r["macro_f1"] - ref["macro_f1"]) < 1e-12 and abs(r["accuracy"] - ref["accuracy"]) < 1e-12
    assert abs(r["recall_Normal"] - ref["recall_Normal"]) < 1e-12


def test_splits_only_trains_nothing_and_writes_summary():
    from src.run_ab import splits_only
    out = Path(tempfile.mkdtemp())
    summary, cuts = splits_only(ARR, AB, ["A", "B"], SEEDS, out, log=lambda *_: None)
    assert len(summary) == (2 + 1) * 5 and not (out / "ab_results_partial.csv").exists()
    assert len(list(out.glob("ab_split_summary_*.csv"))) == 1 and len(list(out.glob("ab_cuts_*.csv"))) == 1
    assert all(r["n_test"] > 0 for r in summary if r["technique"] == "B")
