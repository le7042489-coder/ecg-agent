"""Production NPU anomaly gate — full-spec TSRNet on the Intel NPU via OpenVINO (torch-free).

Only the model forward runs on the NPU; the 3 masked passes + Peak-based Error scoring are numpy.
Device deps: openvino, numpy, scipy, heartpy  (NO torch / .pt / model code). The ecg-bedside conda
env already exports ZE_ENABLE_ALT_DRIVERS so 'NPU' is available.

Calibrate the threshold on TARGET NORMALS (ideally Lepod), then run the gate:
  # 1) calibrate on normals (on-device, on the NPU):
  python ecg_gate_npu.py --onnx tsrnet_spec.onnx --calibrate normals.npy [--cal-labels lab.npy] --thr-out thr.json
  # 2) run the gate on a real Lepod .mat -- slides a 10s window across the WHOLE recording (not just
  #    the first 10s); on a rolling M-of-N wake it saves the abnormal window to a .mat and launches
  #    the real bedside Agent on it:
  python ecg_gate_npu.py --onnx tsrnet_spec.onnx --thr thr.json --mat ecg.mat --hop 2.5 --m 3 --n 5 \
      --wake web   # or: --wake cli   (web=web_server.py, cli=bedside_agent.py; --wake-dry to test wiring)
  # --deploy-dir / --gguf / --host / --port override the launch; --wake-cmd "...{mat}..." for a custom cmd.
  # replay/validate a labelled set (m=n=2 reproduces the old "2 consecutive" debounce):
  python ecg_gate_npu.py --onnx tsrnet_spec.onnx --thr thr.json --npy windows.npy --labels lab.npy --m 2 --n 2
  (--device CPU to compare against CPU)
"""
import argparse, json, os, sys, subprocess
import numpy as np
from scipy.signal import stft
from openvino import Core
import rolling_gate as rg
try:
    import heartpy as hp
except Exception:
    hp = None

MRT, MRS, DIMS = 30, 20, 12
WIN, FS = 5000, 500.0


def rpeaks(lead, fs=500.0):
    if hp is None:
        return np.asarray([])
    try:
        wd, _ = hp.process(np.ascontiguousarray(lead, dtype=float), fs)
        return np.asarray(wd["peaklist"])
    except Exception:
        return np.asarray([])


def spec_of(win4800):
    _, _, Z = stft(win4800.transpose(1, 0), fs=500, window="hann", nperseg=125)
    return np.abs(Z).transpose(1, 2, 0).astype(np.float32)  # (F,T,12)


def load_mat_raw(path):
    """Lepod .mat -> raw (N,12) @500Hz, un-normalized (slide_windows normalizes each window)."""
    import scipy.io
    feats = np.asarray(scipy.io.loadmat(path)["feats"], dtype=float)  # (12,N) @500Hz
    return feats.T


# ----- wake wiring: launch the real bedside Agent on the exact window that tripped the gate -----

def _save_wake_mat(seg, wake_dir, t):
    """Dump the abnormal window as feats (12,WIN) @500Hz so the Agent analyses *what tripped the
    gate* (not the first 10s of a long recording). seg is (N,12) raw mV (or (WIN,12) replay)."""
    import scipy.io
    os.makedirs(wake_dir, exist_ok=True)
    feats = np.asarray(seg, dtype=np.float32)
    if feats.shape[0] != DIMS:           # (N,12) -> (12,N)
        feats = feats.T
    if feats.shape[1] < WIN:
        feats = np.pad(feats, ((0, 0), (0, WIN - feats.shape[1])))
    # absolute: the wake cmd 'cd's into deploy-dir before passing --mat, so a relative path would miss
    path = os.path.abspath(os.path.join(wake_dir, f"wake_{int(round(t))}s.mat"))
    # match lepod2mat.py's .mat schema exactly: feats (12,5000) mV + curr_sample_rate, both required
    # by the bedside Agent's classification tool (it reads ecg['curr_sample_rate'] & ecg['feats'])
    scipy.io.savemat(path, {"feats": feats[:, :WIN], "curr_sample_rate": np.array([[int(FS)]])})
    return path


def build_wake_cmd(args, mat):
    """Real bedside Agent launch for an abnormal-window .mat (matches deploy/dk2500 entrypoints)."""
    py, dd = sys.executable, os.path.expanduser(args.deploy_dir)
    gguf = os.path.expanduser(args.gguf)
    if args.wake == "web":
        return (f"cd {dd} && {py} web_server.py --mat {mat} --backend {args.backend} "
                f"--gguf {gguf} --host {args.host} --port {args.port}")
    return f"cd {dd} && {py} bedside_agent.py --mat {mat} --backend {args.backend} --gguf {gguf}"


def make_on_wake(args, seg_for):
    """on_wake(i,t,window,name): save the abnormal window to a .mat, then launch the Agent on it.
    --wake-cmd (custom, {mat} substituted with the saved window) overrides --wake web|cli.
    --wake-dry prints the command without launching. web mode launches once (one server)."""
    state = {"web_up": False}

    def on_wake(i, t, window, name):
        mat = _save_wake_mat(seg_for(i), os.path.expanduser(args.wake_dir), t)
        if args.wake_cmd:
            cmd = args.wake_cmd.replace("{mat}", mat)
        elif args.wake in ("web", "cli"):
            cmd = build_wake_cmd(args, mat)
        else:
            print(f"    abnormal window saved -> {mat} (no --wake target; not launching)")
            return
        if args.wake == "web" and state["web_up"] and not args.wake_cmd:
            print(f"    web server already up; abnormal window saved -> {mat}")
            return
        log = mat + ".log"
        print(f"    WAKE -> abnormal window {mat}\n    run: {cmd}\n    (Agent stdout -> {log})")
        if args.wake_dry:
            print("    (--wake-dry: not launched)")
        else:
            # fully detached: the gate is a long-running daemon, so the launched Agent must NOT
            # inherit its stdio (that would tether the gate) nor die with it -> own session + logfile
            subprocess.Popen(cmd, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
            state["web_up"] = True

    return on_wake


def make_notify(args, seg_for, thr):
    """Return on_step(i,t,window,name,decision): push live gate status to an always-on web UI, and
    on a wake POST the abnormal window (saved as a .mat) so the page alerts + switches to it.
    Best-effort: a short timeout + swallowed errors so a down/slow server never stalls the gate."""
    import urllib.request
    base = args.notify_url.rstrip("/")

    def post(obj):
        try:
            req = urllib.request.Request(base + "/api/gate", data=json.dumps(obj).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=1.5).read()
        except Exception:
            pass  # server may be down; live status is non-critical

    def on_step(i, t, window, name, d):
        post({"type": "status", "state": d["state"], "score": round(d["score"], 4),
              "smoothed": round(d["smoothed"], 4), "thr": round(thr, 4), "t": round(t, 1),
              "abn": bool(d["abn"]), "count": int(d["count"]), "n": int(d["n"])})
        if d["wake"]:
            mat = _save_wake_mat(seg_for(i), os.path.expanduser(args.wake_dir), t)
            post({"type": "wake", "mat": mat, "score": round(d["score"], 4),
                  "thr": round(thr, 4), "t": round(t, 1)})
            print(f"    WAKE -> notified {base} (abnormal window {mat})")

    return on_step


def score(compiled, ecg5000, r_index):
    """Peak-based Error via NPU forward (full-spec). Higher = more anomalous."""
    t = ecg5000[100:4900, :].astype(np.float32)  # (4800,12)
    sp = spec_of(t)                               # (F,T,12)
    T, nT = t.shape[0], sp.shape[1]
    ml = np.zeros((T, DIMS), bool)
    for r in r_index:
        r = int(r)
        if 200 < r < 4400:
            ml[max(0, r - 240):r + 240, :] = True
    if not ml.any():
        ml[:] = True
    pit, pis = 4800 // MRT, max(1, nT // MRS)
    out = []
    for j in range(100 // MRT):
        mt = t.copy()
        for k in range(MRT):
            c = 48*j + pit*k; mt[c:c+48, :] = 0
        ms = sp.copy()
        for k in range(MRS):
            c = j + pis*k
            if c < nT:
                ms[:, c, :] = 0
        res = compiled({"time": mt[None, ...], "spec": ms[None, ...]})
        recon, var = res[0][0], res[1][0]          # (4800,12), (4800,1)
        l = np.exp(-var) * (recon - t) ** 2
        out.append(float((l * ml).sum() / ml.sum()))
    return float(np.mean(out))


def scores_all(compiled, W):
    return np.array([score(compiled, W[i], rpeaks(W[i][:, 1])) for i in range(len(W))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--calibrate"); ap.add_argument("--cal-labels")
    ap.add_argument("--thr-out", default="gate_threshold_npu.json"); ap.add_argument("--spec-target", type=float, default=0.95)
    ap.add_argument("--thr"); ap.add_argument("--mat"); ap.add_argument("--npy"); ap.add_argument("--labels")
    # rolling window: --hop seconds between re-scores (overlap), M-of-N persistence, median smoothing,
    # refractory seconds between wakes. --debounce d is back-compat sugar for --m d --n d.
    ap.add_argument("--hop", type=float, default=2.5, help="seconds between sliding re-scores (--mat)")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="sleep this many seconds between windows (replay pacing for a live demo; 0=fast)")
    ap.add_argument("--m", type=int); ap.add_argument("--n", type=int)
    ap.add_argument("--smooth", type=int, default=1, help="rolling-median window over raw scores")
    ap.add_argument("--refractory", type=float, default=30.0, help="seconds suppressed after a wake")
    ap.add_argument("--debounce", type=int, default=1)
    # notify an always-on web UI (web_server.py): push live status every window + alert on wake.
    # Independent of --wake (which spawns a fresh process instead).
    ap.add_argument("--notify-url", help="base URL of a running web_server.py, e.g. http://127.0.0.1:8000")
    # wake target: launch the real bedside Agent on the abnormal window's .mat
    ap.add_argument("--wake", choices=["none", "web", "cli"], default="none",
                    help="on abnormal: 'web'=web_server.py, 'cli'=bedside_agent.py, 'none'=just log")
    ap.add_argument("--wake-cmd", help="custom shell cmd ({mat}=saved abnormal window); overrides --wake")
    ap.add_argument("--wake-dry", action="store_true", help="print the wake command without launching")
    ap.add_argument("--deploy-dir", default="~/workspace/ECG-Agent/dk2500_deploy",
                    help="dir holding web_server.py / bedside_agent.py")
    ap.add_argument("--wake-dir", default="wake_mats", help="where to write abnormal-window .mat files")
    ap.add_argument("--gguf", default="~/workspace/ECG-Agent/ecg_agent_llama3b_q4km.gguf")
    ap.add_argument("--backend", default="llama-cpp", choices=["llama-cpp", "transformers"])
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    m = args.m if args.m is not None else args.debounce
    n = args.n if args.n is not None else max(m, args.debounce)

    core = Core()
    print("devices:", core.available_devices)
    compiled = core.compile_model(core.read_model(args.onnx), args.device)
    print("compiled for", args.device)

    if args.calibrate:
        sc = scores_all(compiled, np.load(args.calibrate))
        if args.cal_labels:
            sc = sc[np.load(args.cal_labels).astype(int) == 0]
        thr = float(np.quantile(sc, args.spec_target))
        pcts = {int(p): round(float(np.percentile(sc, p)), 4) for p in (50, 90, 95, 98, 99, 100)}
        json.dump({"threshold": thr, "device": args.device, "n_normal": int(len(sc)),
                   "spec_target": args.spec_target, "score_mean": float(sc.mean()),
                   "score_std": float(sc.std()), "percentiles": pcts}, open(args.thr_out, "w"), indent=2)
        print(f"normal scores: n={len(sc)} mean={sc.mean():.4f} std={sc.std():.4f} pcts(50/90/95/98/99/max)={pcts}")
        print(f"calibrated -> threshold={thr:.6g} @spec={args.spec_target}; saved {args.thr_out}")
        return

    thr = json.load(open(args.thr))["threshold"]
    gate = rg.RollingGate(thr, m=m, n=n, smooth=args.smooth, refractory_s=args.refractory)
    score_fn = lambda w: score(compiled, w, rpeaks(w[:, 1]))
    wake_on = args.wake != "none" or args.wake_cmd
    if args.mat:
        # slide a 10s window across the whole recording at --hop seconds
        raw = load_mat_raw(args.mat)
        triples = list(rg.slide_windows(raw, fs=FS, win=WIN, hop=int(args.hop * FS)))
        starts = [s for s, _, _ in triples]
        times, windows = [t for _, t, _ in triples], [w for _, _, w in triples]
        names = [args.mat] * len(windows)
        # hand the Agent the RAW mV slice of the abnormal window (not the [-1,1]-normalized one)
        seg_for = lambda i: raw[starts[i]:starts[i] + WIN]
        on_wake = make_on_wake(args, seg_for) if wake_on else None
        on_step = make_notify(args, seg_for, thr) if args.notify_url else None
        print(f"sliding {len(windows)} windows (hop={args.hop}s) over {args.mat}; m-of-n={m}/{n}"
              + (f"; wake={args.wake}" if wake_on else "") + (f"; notify={args.notify_url}" if args.notify_url else ""))
        rg.run_stream(windows, times, score_fn, gate, names=names, wake_cmd=args.wake_cmd,
                      on_wake=on_wake, on_step=on_step, pace=args.pace)
    else:
        # pre-cut disjoint windows: time-stamp them one window-length (10s) apart
        windows = list(np.load(args.npy))
        lab = np.load(args.labels) if args.labels else None
        times = [i * (WIN / FS) for i in range(len(windows))]
        seg_for = lambda i: windows[i]
        on_wake = make_on_wake(args, seg_for) if wake_on else None
        on_step = make_notify(args, seg_for, thr) if args.notify_url else None
        print(f"replay {len(windows)} windows; m-of-n={m}/{n} smooth={args.smooth} refractory={args.refractory}s")
        rg.run_stream(windows, times, score_fn, gate, labels=lab, wake_cmd=args.wake_cmd,
                      on_wake=on_wake, on_step=on_step, pace=args.pace)


if __name__ == "__main__":
    main()
