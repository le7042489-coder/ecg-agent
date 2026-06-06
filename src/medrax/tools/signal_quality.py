"""ECG signal-quality assessment tool.

The per-lead acceptable / unacceptable logic vendors the three signal-quality
criteria — (I) stationarity, (II) heart-rate plausibility, (III) in-band SNR —
from the ECGAssess toolbox:

    Kramer L. et al., "ECGAssess: A Python-Based Toolbox to Assess ECG Lead
    Signal Quality", Frontiers in Digital Health, 2022.
    Source: https://github.com/LinusKra/ECGAssess  (Code/AlgorithmsV5.py)

NOTE on licensing: the upstream ECGAssess repository ships WITHOUT a LICENSE file
(i.e. all rights reserved by default). This adaptation is vendored for internal
research use with attribution; confirm licensing with the original authors before
any redistribution.

Adaptations vs. the original GUI code (AlgorithmsV5.py):
  * sampling rate fully parameterized — the Butterworth filters, the stationarity
    window length, and the SNR periodogram band are all computed from the actual
    `fs` instead of the hardcoded 500 Hz / `band*10` bin indexing, which silently
    assumed a 0.1 Hz periodogram resolution (i.e. exactly 10 s recordings);
  * input is an (8, N) physical-lead array + `fs`, not the GUI's global `data[lead]`
    array (where `data[0]` was a skipped time column);
  * GUI residue removed (the `bigO` timing import, `time` prints, and the unicode
    tick/cross result symbols);
  * structured dict output instead of a unicode matrix.

In addition to the (vendored) ECGAssess verdict, this tool reports a continuous
NeuroKit2 `ecg_quality` score (averageQRS method, 0–1) per lead as an auxiliary
signal, and a summary (overall quality + electrodes worth checking).
"""

from typing import Any, Dict, List, Optional, Type

import os
import traceback

import numpy as np
import scipy.io
import scipy.signal
import neurokit2 as nk
from ecgdetectors import Detectors

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


# ─── ECGAssess validated parameters (from AlgorithmsV5.py) ────────────────────
MAX_LOSS_PASSBAND = 0.1     # dB  (Butterworth design)
MIN_LOSS_STOPBAND = 20      # dB
SNR_THRESHOLD = 0.5         # in-band / out-of-band power ratio below this => noisy
SIGNAL_FREQ_BAND = [2, 40]  # Hz, the ECG "signal" band for the SNR check
HEART_RATE_LIMITS = [24, 300]   # bpm, plausible range
WINDOW_SECONDS = 0.2        # stationarity window = 100 samples @ 500 Hz = 0.2 s
HF_FILTER_EDGES = (20, 30)  # Hz, high-frequency-noise low-pass (pass, stop)
BASELINE_FILTER_EDGES = (0.5, 8)  # Hz, baseline-wander low-pass (pass, stop)

# Physical electrode leads (skip the derived III/aVR/aVL/aVF, which are linear
# combinations and carry no independent electrode information).
PHYSICAL_LEAD_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]
PHYSICAL_LEAD_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# Which physical electrode(s) each lead depends on — used to point the bedside
# user at the electrode(s) worth re-checking when a lead is flagged unacceptable.
LEAD_ELECTRODES = {
    "I": ["RA", "LA"], "II": ["RA", "LL"],
    "V1": ["V1"], "V2": ["V2"], "V3": ["V3"],
    "V4": ["V4"], "V5": ["V5"], "V6": ["V6"],
}


# ─── vendored ECGAssess criteria (refactored: per-lead, fs-parameterized) ─────
def _hf_noise_filter(sig: np.ndarray, fs: int) -> np.ndarray:
    """Low-pass to remove high-frequency noise (ECGAssess high_frequency_noise_filter)."""
    order, wn = scipy.signal.buttord(
        HF_FILTER_EDGES[0], HF_FILTER_EDGES[1], MAX_LOSS_PASSBAND, MIN_LOSS_STOPBAND, fs=fs
    )
    b, a = scipy.signal.butter(order, wn, fs=fs)
    return scipy.signal.filtfilt(b, a, sig)


def _baseline_filter(sig: np.ndarray, fs: int) -> np.ndarray:
    """Low-pass that isolates baseline wander (ECGAssess baseline_filter)."""
    order, wn = scipy.signal.buttord(
        BASELINE_FILTER_EDGES[0], BASELINE_FILTER_EDGES[1], MAX_LOSS_PASSBAND, MIN_LOSS_STOPBAND, fs=fs
    )
    b, a = scipy.signal.butter(order, wn, fs=fs)
    return scipy.signal.filtfilt(b, a, sig)


def _stationary_fail(sig: np.ndarray, fs: int) -> bool:
    """Criterion I — flat-line / saturation check.

    True (fail) if any analysis window is perfectly flat (max == min).
    """
    win = max(2, int(round(WINDOW_SECONDS * fs)))
    if len(sig) < win:
        return False  # too short to assess; do not flag
    windows = np.lib.stride_tricks.sliding_window_view(sig, win)[::10]
    for w in windows:
        if np.amax(w) == np.amin(w):
            return True
    return False


def _heart_rate_fail(sig: np.ndarray, fs: int) -> bool:
    """Criterion II — heart-rate plausibility via Pan-Tompkins beat count.

    True (fail) if the detected beat count implies a rate outside 24–300 bpm.
    Operates on the band-pass-filtered signal (as in ECGAssess).
    """
    detectors = Detectors(int(fs))
    beats = detectors.pan_tompkins_detector(sig)
    duration_s = len(sig) / float(fs)
    upper = HEART_RATE_LIMITS[1] * duration_s / 60.0
    lower = HEART_RATE_LIMITS[0] * duration_s / 60.0
    return len(beats) > upper or len(beats) < lower


def _snr_fail(sig: np.ndarray, fs: int) -> bool:
    """Criterion III — in-band signal-to-noise ratio.

    SNR = power in [2, 40] Hz / power outside that band. True (fail) if < 0.5.

    The original indexed the periodogram as `pxx[band*10]`, hardcoding a 0.1 Hz
    bin resolution (valid only for exactly-10 s, 500 Hz recordings). Here the band
    is selected from the actual frequency axis returned by `periodogram`.
    """
    f, pxx = scipy.signal.periodogram(sig, fs=fs, scaling="spectrum")
    total_power = float(np.sum(pxx))
    if total_power <= 0:
        return False  # no power at all; matches the original "else -> pass" branch
    band_mask = (f >= SIGNAL_FREQ_BAND[0]) & (f < SIGNAL_FREQ_BAND[1])
    signal_power = float(np.sum(pxx[band_mask]))
    noise_power = total_power - signal_power
    if noise_power <= 0:
        return False
    return (signal_power / noise_power) < SNR_THRESHOLD


def _snr_noise_breakdown(sig: np.ndarray, fs: int):
    """诊断 SNR 失败主因（仅描述，不参与 acceptable 判定）。

    返回 (<2Hz 基线占总功率比, ≥40Hz 高频/工频占总功率比)。periodogram 默认去均值，
    故 <2Hz 即基线漂移、≥40Hz 即高频/工频干扰。
    """
    f, pxx = scipy.signal.periodogram(sig, fs=fs, scaling="spectrum")
    total = float(np.sum(pxx))
    if total <= 0:
        return 0.0, 0.0
    lt2 = float(np.sum(pxx[f < SIGNAL_FREQ_BAND[0]])) / total
    gt40 = float(np.sum(pxx[f >= SIGNAL_FREQ_BAND[1]])) / total
    return lt2, gt40


def _quality_hint(snr_lt2: List[float], snr_gt40: List[float]) -> str:
    """据 SNR 失败导联的噪声构成给出可操作的失败主因提示（不影响 acceptable 判定）。"""
    if not snr_lt2:
        return ""
    mean_lt2 = sum(snr_lt2) / len(snr_lt2)
    mean_gt40 = sum(snr_gt40) / len(snr_gt40)
    if mean_lt2 >= 0.5:
        return ("信号以低频基线漂移为主（<2Hz 占比高）：建议检查电极是否贴牢、"
                "保持静止、减少呼吸/移动伪迹后重测。")
    if mean_gt40 >= 0.3:
        return ("信号以高频/工频干扰为主（≥40Hz 占比高）：建议远离电源线、"
                "检查接地与导联线屏蔽后重测。")
    return "信号整体信噪比偏低（宽带噪声）：建议改善电极接触、保持静止后重新采集。"


def _nk_quality(sig: np.ndarray, fs: int) -> Optional[float]:
    """Auxiliary continuous quality score (NeuroKit2 averageQRS, mean over the lead)."""
    try:
        cleaned = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
        q = np.asarray(nk.ecg_quality(cleaned, sampling_rate=fs, method="averageQRS"), dtype=float)
        q = q[~np.isnan(q)]
        if q.size == 0:
            return None
        return round(float(np.clip(np.mean(q), 0.0, 1.0)), 3)
    except Exception:
        return None


class ECGSignalQualityInput(BaseModel):
    """Input for the ECG signal-quality tool. Only supports MAT files."""

    ecg_path: str = Field(
        ..., description="Path to the ECG signal file, only supports MAT files"
    )


class ECGSignalQualityTool(BaseTool):
    """Tool that assesses per-lead ECG signal quality (acceptable / unacceptable).

    For each physical electrode lead (I, II, V1–V6; the derived limb leads
    III/aVR/aVL/aVF are skipped) it applies the three ECGAssess criteria
    (stationarity, heart-rate plausibility, in-band SNR) and reports whether the
    lead is acceptable plus which criteria failed. It additionally reports a
    continuous NeuroKit2 quality score per lead, and a summary: an overall quality
    verdict and the electrodes worth re-checking.

    This is a recording-quality / electrode-contact tool — it answers "is this ECG
    clean / trustworthy / are any leads bad", NOT clinical questions. It makes no
    diagnostic or morphological determination.
    """

    name: str = "ecg_signal_quality"
    description: str = (
        "A tool that assesses ECG recording/signal quality per lead. For each physical "
        "lead (I, II, V1-V6) it returns acceptable/unacceptable and the failed criteria "
        "(stationarity = flat/saturated, heart_rate = implausible rate, snr = too noisy), "
        "a continuous NeuroKit quality score (0-1), and a summary (overall_quality and "
        "suspect_electrodes worth re-checking). Input is the path to an ECG .mat file. "
        "Use this tool for questions about whether the recording is clean / reliable, "
        "whether any leads are noisy or disconnected, or which electrode to check. "
        "It does NOT diagnose cardiac conditions or measure waveform morphology."
    )
    args_schema: Type[BaseModel] = ECGSignalQualityInput
    _device: Optional[str] = PrivateAttr(default="cpu")

    def __init__(self, device: Optional[str] = "cpu"):
        """Initialize the signal-quality tool.

        Args:
            device: Unused (kept for a uniform tool interface); runs on CPU (scipy/neurokit).
        """
        super().__init__()
        self._device = device

    def _process_ecg_mat(self, ecg_path: str):
        """Load (signal[12,N], sampling_rate) from a MAT file."""
        ecg_data = scipy.io.loadmat(ecg_path)
        signal = None
        if "feats" in ecg_data:
            signal = ecg_data["feats"]
        else:
            for key in ["data", "ECG", "signal", "val"]:
                if key in ecg_data:
                    signal = ecg_data[key]
                    break
        if signal is None:
            raise ValueError(f"Could not find ECG data in MAT file: {ecg_path}")

        fs = 500
        if "curr_sample_rate" in ecg_data:
            try:
                fs = int(np.asarray(ecg_data["curr_sample_rate"]).flatten()[0])
            except Exception:
                fs = 500
        return np.asarray(signal, dtype=np.float64), fs

    def _assess(self, signal: np.ndarray, fs: int) -> Dict[str, Any]:
        """Run ECGAssess + NeuroKit per physical lead and summarize."""
        n_leads = signal.shape[0]
        if n_leads < 12:
            return {
                "analysis_status": "failed",
                "note": f"Expected a 12-lead signal; got {n_leads} leads.",
            }

        leads: Dict[str, Any] = {}
        unacceptable: Dict[str, List[str]] = {}
        suspect_electrodes: List[str] = []
        snr_lt2: List[float] = []   # SNR 失败导联的 <2Hz 基线占比（用于失败主因提示）
        snr_gt40: List[float] = []  # SNR 失败导联的 ≥40Hz 高频/工频占比

        for idx, name in zip(PHYSICAL_LEAD_INDICES, PHYSICAL_LEAD_NAMES):
            raw = signal[idx, :]
            failed: List[str] = []

            # A dead / non-finite lead is unacceptable outright.
            if not np.all(np.isfinite(raw)) or np.all(raw == raw.flat[0]):
                failed = ["invalid_signal"]
            else:
                try:
                    if _stationary_fail(raw, fs):
                        failed.append("stationarity")
                except Exception:
                    pass
                try:
                    filt = _hf_noise_filter(raw, fs) - _baseline_filter(raw, fs)
                    if _heart_rate_fail(filt, fs):
                        failed.append("heart_rate")
                except Exception:
                    pass
                try:
                    if _snr_fail(raw, fs):
                        failed.append("snr")
                        lt2, gt40 = _snr_noise_breakdown(raw, fs)
                        snr_lt2.append(lt2)
                        snr_gt40.append(gt40)
                except Exception:
                    pass

            acceptable = len(failed) == 0
            leads[name] = {
                "acceptable": acceptable,
                "failed_criteria": failed,
                "nk_quality": _nk_quality(raw, fs),
            }
            if not acceptable:
                unacceptable[name] = failed
                for e in LEAD_ELECTRODES.get(name, []):
                    if e not in suspect_electrodes:
                        suspect_electrodes.append(e)

        n_bad = len(unacceptable)
        if n_bad == 0:
            overall = "all_acceptable"
        elif n_bad == len(PHYSICAL_LEAD_NAMES):
            overall = "all_unacceptable"
        else:
            overall = "partial"

        quality_hint = _quality_hint(snr_lt2, snr_gt40)

        return {
            "sampling_rate": fs,
            "overall_quality": overall,
            "acceptable_lead_count": len(PHYSICAL_LEAD_NAMES) - n_bad,
            "total_leads_assessed": len(PHYSICAL_LEAD_NAMES),
            "leads": leads,
            "unacceptable_leads": unacceptable,
            "suspect_electrodes": sorted(suspect_electrodes),
            "quality_hint": quality_hint,
            "note": (
                "Per-lead recording-quality assessment (ECGAssess criteria: stationarity / "
                "heart_rate / snr) plus a NeuroKit averageQRS score (0-1). Physical leads "
                "only (I, II, V1-V6). This reflects signal quality, not clinical findings."
            ),
            "analysis_status": "completed",
        }

    def _run(
        self,
        ecg_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Assess per-lead signal quality for an ECG .mat file.

        Args:
            ecg_path: Absolute path to the ECG signal file (.mat format).
            run_manager: Optional callback manager for the tool run.

        Returns:
            Dict[str, Any]: Per-lead quality verdicts, NeuroKit scores, and a summary.
        """
        print(f"Using ECG file for signal-quality analysis: {ecg_path}")

        if not os.path.exists(ecg_path):
            return {
                "error": f"ECG file not found: {ecg_path}",
                "analysis_status": "failed",
                "note": "Could not locate the specified ECG file",
            }

        try:
            signal, fs = self._process_ecg_mat(ecg_path)
            result = self._assess(signal, fs)
            result["ecg_path"] = ecg_path
            return result
        except Exception as e:
            traceback.print_exc()
            return {
                "error": str(e),
                "ecg_path": ecg_path,
                "analysis_status": "failed",
                "note": f"Signal-quality analysis failed due to: {str(e)}",
            }

    async def _arun(
        self,
        ecg_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Asynchronously assess signal quality (delegates to the synchronous implementation)."""
        return self._run(ecg_path)
