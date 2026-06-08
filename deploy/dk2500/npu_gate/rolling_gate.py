"""Rolling-window decision core for the anomaly gate (pure numpy, torch/OpenVINO-free).

The scorer (time-only / full-spec TSRNet) takes a fixed 10 s window and emits one Peak-based
Error score. This module turns that into a *continuous monitor*:

  1) `slide_windows` — chop a continuous recording (or a live ring buffer) into OVERLAPPING 10 s
     windows at a hop < window, so we re-score every `hop` seconds instead of once per disjoint
     block. Lower miss latency + more votes per episode.
  2) `RollingGate` — a streaming state machine that decides when to WAKE the Agent from the stream
     of scores. It replaces the brittle "N consecutive abnormal windows" debounce with:
        - optional rolling-median smoothing (kills single-window motion-artifact spikes),
        - an M-of-N persistence rule (wake when >= M of the last N windows are abnormal; tolerant of
          one clean window inside a sustained run, which is common with overlapping windows),
        - a refractory + re-arm hysteresis (one wake per episode; don't spam the Agent while the
          same abnormal episode persists, and require the buffer to calm before re-arming).
  3) `run_stream` — drives a window iterator through a `score_fn` + a `RollingGate`, prints per
     window, fires the wake-cmd, and (for labelled replay) reports AUC/sens/spec. Both the torch
     prototype (`ecg_gate.py`) and the NPU production gate (`ecg_gate_npu.py`) share this so the
     decision logic is identical on dev and on-device.

Why this matters here: Lepod normals sit at score ~0.22 (thr ~0.28) — a tight margin, so isolated
normal windows spike over threshold. M-of-N + median smoothing suppress those false wakes while the
overlap keeps real episodes caught quickly. See memory project-npu-gate-tsrnet.
"""
from collections import deque
import subprocess
import numpy as np

WIN = 5000  # 10 s @ 500 Hz — the scorer's fixed input length


def norm_window(w):
    """(L,12) raw mV -> per-lead min-max [-1,1] (same scheme as preprocess/utils.normalize)."""
    mn, mx = w.min(0, keepdims=True), w.max(0, keepdims=True)
    return (2 * (w - mn) / (mx - mn + 1e-8) - 1).astype(np.float32)


def slide_windows(signal, fs=500.0, win=WIN, hop=None):
    """Segment a continuous (N,12) recording into overlapping, min-max-normalized `win`-sample
    windows. Yields (start_sample, t_seconds, norm_window). hop defaults to win//4 (75% overlap).

    Shorter-than-win recordings -> a single zero-padded window. A final right-aligned window is
    appended when the regular grid leaves an uncovered tail, so the end of the record is scored.
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.shape[0]
    hop = win // 4 if hop is None else int(hop)
    if n <= win:
        w = np.pad(signal, ((0, win - n), (0, 0))) if n < win else signal
        yield 0, 0.0, norm_window(w)
        return
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] != n - win:  # cover the trailing remainder
        starts.append(n - win)
    for s in starts:
        yield s, s / fs, norm_window(signal[s:s + win])


class RollingGate:
    """Streaming wake decision over a rolling window of recent scores.

    Per `update(score, t)`:
      smoothed = median(last `smooth` raw scores);  abn = smoothed >= threshold
      count    = # abnormal in the last `n` windows
      WAKE     when ARMED and count >= `m`  ->  enter refractory
      re-arm   when refractory elapsed (>= `refractory_s`) AND count < `m` (signal calmed)

    The classic "k consecutive abnormal" debounce is the special case m=n=k (with refractory_s=0).
    """

    def __init__(self, threshold, m=3, n=5, smooth=1, refractory_s=30.0):
        if n < 1 or m < 1 or m > n:
            raise ValueError(f"need 1 <= m <= n, got m={m} n={n}")
        self.thr = float(threshold)
        self.m, self.n = int(m), int(n)
        self.smooth = max(1, int(smooth))
        self.refractory_s = float(refractory_s)
        self.flags = deque(maxlen=self.n)     # recent abnormal booleans
        self.raw = deque(maxlen=self.smooth)  # recent raw scores (for median smoothing)
        self.armed = True
        self.last_wake_t = None

    def update(self, score, t):
        """Feed one window score at time `t` (seconds). Returns a decision dict; wake=True on a
        fresh wake event for this window."""
        self.raw.append(float(score))
        smoothed = float(np.median(self.raw))
        abn = smoothed >= self.thr
        self.flags.append(bool(abn))
        count = sum(self.flags)

        if not self.armed:  # in refractory: re-arm once time has passed and the buffer calmed
            if self.last_wake_t is not None and (t - self.last_wake_t) >= self.refractory_s \
                    and count < self.m:
                self.armed = True

        wake = False
        if self.armed and count >= self.m:
            wake, self.armed, self.last_wake_t = True, False, t

        return {"t": t, "score": float(score), "smoothed": smoothed, "abn": abn,
                "count": count, "n": self.n, "m": self.m, "state": "armed" if self.armed else "refractory",
                "wake": wake}


def run_stream(windows, times, score_fn, gate, labels=None, names=None, wake_cmd=None,
               on_wake=None, verbose=True):
    """Drive (start, t, window) items through score_fn + gate. `windows`/`times` are parallel lists
    (or an iterable of (t, window)); score_fn(window5000)->float. Returns the np.array of scores.

    On a wake event: if `on_wake` is given it is called as on_wake(i, t, window, name) (the caller
    decides what to launch -- e.g. dump the abnormal window to a .mat and start the bedside Agent);
    otherwise, if `wake_cmd` is given, it is run with {mat} substituted by names[i]. For labelled
    replay (`labels` set) prints an AUC/sens/spec summary at the end.
    """
    scores, woke = [], 0
    for i, (t, w) in enumerate(zip(times, windows)):
        s = float(score_fn(w))
        d = gate.update(s, t)
        scores.append(s)
        if verbose:
            gt = f" gt={int(labels[i])}" if labels is not None else ""
            tag = "ABN" if d["abn"] else "norm"
            head = f"[{i}] t={t:6.1f}s score={s:.4g} sm={d['smoothed']:.4g} {tag}{gt} ({d['count']}/{d['n']})"
            if d["wake"]:
                print(f"{head} >>> WAKE")
            else:
                print(f"{head} {d['state']}")
        if d["wake"]:
            woke += 1
            name = names[i] if names else None
            if on_wake is not None:
                on_wake(i, t, w, name)
            elif wake_cmd:
                cmd = wake_cmd.replace("{mat}", name or "")
                print("    run:", cmd)
                subprocess.Popen(cmd, shell=True)
    scores = np.asarray(scores)
    if labels is not None and verbose:
        labels = np.asarray(labels)
        thr = gate.thr
        line = f"\nreplay: windows={len(scores)} wakes={woke} thr={thr:.4g}"
        try:
            from sklearn.metrics import roc_auc_score
            line += (f"  AUC={roc_auc_score(labels, scores):.4f}"
                     f"  sens={(scores[labels==1]>=thr).mean():.3f}"
                     f"  spec={(scores[labels==0]<thr).mean():.3f}")
        except Exception:
            line += f"  fired={(scores>=thr).sum()} (sklearn absent -> no AUC)"
        print(line)
    return scores
