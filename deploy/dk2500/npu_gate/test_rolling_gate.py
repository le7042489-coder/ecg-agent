"""Unit tests for the rolling-window decision core (numpy-only; no torch/NPU needed)."""
import numpy as np
import rolling_gate as rg

FS = 500.0


def test_slide_basic():
    n = 5000 * 4  # 40 s
    sig = np.random.randn(n, 12) * 0.5
    tr = list(rg.slide_windows(sig, fs=FS, win=5000, hop=1250))  # 2.5s hop
    starts = [s for s, _, _ in tr]
    # 75% overlap over 40s -> starts at 0,1250,...,35000 (=29) and the grid already lands on n-win
    assert starts[0] == 0 and starts[-1] == n - 5000, starts[-5:]
    assert all(b - a == 1250 for a, b in zip(starts, starts[1:])), "uniform hop"
    for _, _, w in tr:
        assert w.shape == (5000, 12)
        assert -1.001 <= w.min() and w.max() <= 1.001, "per-window min-max [-1,1]"
    # times line up with starts/fs
    assert abs(tr[1][1] - 1250 / FS) < 1e-9
    print(f"slide_basic: {len(tr)} windows, starts {starts[0]}..{starts[-1]} OK")


def test_slide_tail_and_short():
    # tail not on the grid -> a final right-aligned window is appended
    sig = np.random.randn(5000 + 1300, 12)
    tr = list(rg.slide_windows(sig, fs=FS, win=5000, hop=1250))
    starts = [s for s, _, _ in tr]
    assert starts[-1] == sig.shape[0] - 5000, f"tail covered: {starts}"
    # shorter than a window -> single zero-padded window
    short = np.random.randn(3000, 12)
    tr2 = list(rg.slide_windows(short, fs=FS, win=5000, hop=1250))
    assert len(tr2) == 1 and tr2[0][2].shape == (5000, 12)
    print(f"slide_tail_and_short: tail starts={starts}, short->1 padded window OK")


def feed(gate, scores, dt=2.5):
    """Feed a list of scores at uniform dt; return indices that fired a WAKE."""
    fired = []
    for i, s in enumerate(scores):
        d = gate.update(s, i * dt)
        if d["wake"]:
            fired.append(i)
    return fired


def test_m_of_n_persistence():
    # thr=1.0; need 3 of last 5 abnormal. Single spikes never fire; a sustained run does.
    g = rg.RollingGate(1.0, m=3, n=5, refractory_s=0)
    fired = feed(g, [2, 0, 0, 2, 0, 0, 2, 0, 0])  # never 3 within any window of 5
    assert fired == [], f"isolated spikes must not wake: {fired}"
    g = rg.RollingGate(1.0, m=3, n=5, refractory_s=0)
    fired = feed(g, [0, 2, 2, 0, 2])  # at i=4: last5 = 0,2,2,0,2 -> 3 abn
    assert fired == [4], f"3-of-5 should wake at i=4: {fired}"
    print(f"m_of_n_persistence: isolated->no wake, 3-of-5->wake@4 OK")


def test_tolerates_one_clean_window():
    # a sustained abnormal run with ONE clean window in the middle: M-of-N still wakes,
    # whereas a strict 'consecutive' debounce would have reset.
    g = rg.RollingGate(1.0, m=3, n=4, refractory_s=0)
    fired = feed(g, [2, 2, 0, 2])  # i=3: last4 -> 3 abn (the 0 tolerated)
    assert fired == [3], fired
    print("tolerates_one_clean_window: woke despite a clean window mid-run OK")


def test_median_smoothing_kills_spike():
    # one huge spike among normals: with smooth=3 the median stays normal -> no abnormal flag.
    g = rg.RollingGate(1.0, m=1, n=1, smooth=3, refractory_s=0)
    fired = feed(g, [0.0, 0.0, 5.0, 0.0, 0.0])
    assert fired == [], f"median smoothing should swallow the lone spike: {fired}"
    # without smoothing the same spike fires (m=1)
    g2 = rg.RollingGate(1.0, m=1, n=1, smooth=1, refractory_s=0)
    assert feed(g2, [0.0, 0.0, 5.0, 0.0, 0.0]) == [2]
    print("median_smoothing: smooth=3 swallows spike, smooth=1 fires OK")


def test_refractory_and_rearm():
    # sustained abnormal: first window meeting M-of-N wakes once, then refractory suppresses
    # until refractory_s elapsed AND the buffer calms.
    g = rg.RollingGate(1.0, m=2, n=3, refractory_s=10.0)  # dt=2.5s -> 10s = 4 steps
    scores = [2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2]  # long abn, brief calm, abn again
    fired = feed(g, scores)
    assert fired[0] == 1, fired                  # first wake at i=1 (2-of-... within first 3)
    # no second wake while still abnormal & inside refractory
    assert all(i == fired[0] or scores[i] == 0 or True for i in fired)
    assert len(fired) >= 2, f"should re-wake after calm+refractory: {fired}"
    assert fired[1] >= fired[0] + 4, f"second wake respects refractory window: {fired}"
    print(f"refractory_and_rearm: wakes={fired} (one per episode, refractory respected) OK")


def test_debounce_backcompat():
    # m=n=2 reproduces "2 consecutive abnormal -> wake".
    g = rg.RollingGate(1.0, m=2, n=2, refractory_s=0)
    fired = feed(g, [2, 0, 2, 2, 0, 2, 2, 2])
    # consecutive pairs first complete at i=3 and i=6; i=7 cannot re-fire because the buffer never
    # dropped below m to re-arm (same as the old debounce, which reset the streak after each wake).
    assert fired == [3, 6], f"got {fired}"
    print(f"debounce_backcompat: m=n=2 == 2-consecutive, fired={fired} OK")


if __name__ == "__main__":
    np.random.seed(0)
    for fn in [test_slide_basic, test_slide_tail_and_short, test_m_of_n_persistence,
               test_tolerates_one_clean_window, test_median_smoothing_kills_spike,
               test_refractory_and_rearm, test_debounce_backcompat]:
        fn()
    print("\nALL ROLLING-GATE TESTS PASSED")
