import numpy as np

from src.common import CLASSES
from src.data_ab import prepare
from src.lookup import LookupModel
from src.splits_ab import burst_cut, seen_mask, split_a, split_b, train_from_test
from src.synthetic_ab import make_recordings

DF, META = make_recordings(seed=1)
ARR = prepare(DF, sanitise=True, tail_gap=1.0)


def test_sanitise_removes_exactly_the_tails():
    assert ARR["removed"] == {s: META["tail_frames"] for s in ["dos", "fuzzy", "gear", "rpm"]}
    assert len(ARR["y"]) == len(DF) - 4 * META["tail_frames"]
    assert "normal" not in ARR["removed"]


def test_stride_is_applied_after_sanitising():
    small = prepare(DF, sanitise=True, tail_gap=1.0, stride=10)
    assert abs(len(small["y"]) - len(ARR["y"]) / 10) <= 1


def test_split_a_exact_counts_complete_and_seeded():
    y = ARR["y"]
    t42 = split_a(y, 42)
    for c in range(5):
        assert int((y[t42] == c).sum()) == round(0.2 * int((y == c).sum()))
    assert len(np.unique(t42)) == len(t42)
    assert (split_a(y, 42) == t42).all()
    assert not (split_a(y, 43) == t42).all()
    tr = train_from_test(len(y), t42)
    assert len(tr) + len(t42) == len(y) and len(np.intersect1d(tr, t42)) == 0


def test_burst_cut_hand_made_cases():
    ts = np.array([0, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40.0])
    is_t = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    c = burst_cut(ts, is_t)                       # tau = 32 is the end of the last burst -> quiet after it
    assert c["n_bursts"] == 3 and abs(c["cut"] - 36.0) < 1e-9 and abs(c["quiet_len"] - 8.0) < 1e-9
    ts2 = ts.copy(); ts2[-1] = 50.0               # tau = 40 lies in the quiet stretch after the last burst
    assert abs(burst_cut(ts2, is_t)["cut"] - 41.0) < 1e-9
    ts3 = np.array([0, 10, 11, 12, 20, 21, 22, 27.5])         # tau = 22 is the end of the last burst -> 22 to 27.5
    is3 = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    assert abs(burst_cut(ts3, is3)["cut"] - 24.75) < 1e-9
    t_frames = [10, 11, 12] + list(range(40, 47)) + [50, 51, 52]    # three bursts, frames 1 s apart
    ts4 = np.array([0.0] + t_frames + [55.0])                       # tau = 44 is inside the middle burst (40 to 46)
    is4 = np.array([False] + [True] * len(t_frames) + [False])
    c4 = burst_cut(ts4, is4)
    assert c4["n_bursts"] == 3 and abs(c4["cut"] - 48.0) < 1e-9 and abs(c4["quiet_len"] - 4.0) < 1e-9
    ts5 = np.array([0.0] + t_frames + [100.0])                      # tau = 80 lies in the long quiet stretch after the last burst
    assert abs(burst_cut(ts5, is4)["cut"] - 76.0) < 1e-9
    amb = burst_cut(np.linspace(0, 100, 11), np.zeros(11, bool))   # no T frames: plain 80% point
    assert abs(amb["cut"] - 80.0) < 1e-9 and amb["n_bursts"] == 0


def test_split_b_train_before_test_and_bursts_intact():
    test_idx, info = split_b(ARR["ts"], ARR["is_t"], ARR["source"])
    is_test = np.zeros(len(ARR["y"]), bool)
    is_test[test_idx] = True
    names = ARR["source_names"]
    for s, c in info.items():
        m = ARR["source"] == s
        ts = ARR["ts"][m]
        assert ts[~is_test[m]].max() <= c["cut"] < ts[is_test[m]].min()          # training entirely before testing
        assert c["n_bursts"] == (0 if names[s] == "normal" else META["bursts"])
        if names[s] != "normal":
            tt = ts[ARR["is_t"][m]]
            burst_id = np.concatenate([[0], np.cumsum(np.diff(tt) > 1.0)])
            side = is_test[m][ARR["is_t"][m]]
            for b in range(META["bursts"]):
                assert len(set(side[burst_id == b])) == 1                       # a burst is never cut
            assert abs(c["quiet_len"] - 3.0) < 0.05
        else:
            assert abs(c["cut_rel"] - 0.8 * (c["t1"] - c["t0"])) < 1e-9
    assert len(test_idx) + len(train_from_test(len(ARR["y"]), test_idx)) == len(ARR["y"])   # no frame discarded
    for c in range(1, 5):                                                       # every attack class is on both sides
        assert 0 < (ARR["y"][test_idx] == c).sum() < (ARR["y"] == c).sum()


def test_seen_mask_matches_brute_force():
    rng = np.random.default_rng(0)
    sig = rng.integers(0, 50, size=400).astype(np.int32)
    test = np.sort(rng.choice(400, size=80, replace=False))
    train = train_from_test(400, test)
    brute = np.array([s in set(sig[train].tolist()) for s in sig[test]])
    assert (seen_mask(sig, train, test) == brute).all()


def test_lookup_matches_brute_force_and_falls_back_to_normal():
    rng = np.random.default_rng(1)
    sig = rng.integers(0, 40, size=300).astype(np.int32)
    y = (sig % 4 + (rng.random(300) < 0.05)).clip(0, 4).astype(np.int8)    # mostly one label per signature
    tr, te = np.arange(0, 200), np.arange(200, 300)
    pred = LookupModel(0).fit(sig[tr], y[tr]).predict(sig[te])
    for s, p in zip(sig[te], pred):
        labs = y[tr][sig[tr] == s]
        want = np.bincount(labs, minlength=5).argmax() if len(labs) else 0
        assert p == want
    assert LookupModel(0).fit(sig[tr], y[tr]).predict(np.array([10_000], dtype=np.int32))[0] == 0   # unseen -> Normal
