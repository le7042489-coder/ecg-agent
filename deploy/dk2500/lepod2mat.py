"""
Lepod Pro (ER3) CSV → PTB-XL .mat 格式转换器

输入：lepod_ecg.py 输出的 CSV（8通道, 250Hz, mV）
输出：ECG-Agent 兼容的 .mat 文件（12导联, 500Hz, 10秒, mV）

12 导联重建：
  Lepod 输出 8 通道: V6, I, II, V1, V2, V3, V4, V5
  导出标准 12 导联 (PTB-XL 顺序):
    [0] I    [1] II   [2] III=II-I   [3] aVR=-(I+II)/2
    [4] aVL=I-II/2    [5] aVF=II-I/2
    [6] V1  [7] V2   [8] V3  [9] V4  [10] V5  [11] V6
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal


TARGET_SAMPLE_RATE = 500   # fairseq-signals ECGTransformer 期望 500Hz
TARGET_SAMPLES = 5000      # 10 秒 × 500Hz


def load_lepod_csv(csv_path: str) -> tuple[np.ndarray, list[str]]:
    """读取 lepod_ecg.py 输出的 CSV。

    返回 (signal, channel_names)
    signal shape: (n_samples, 8)，单位 mV
    """
    df = pd.read_csv(csv_path)

    # lepod_ecg.py 的列命名规则：最后 8 列是通道数据
    # 典型列名候选：V6, I, II, V1, V2, V3, V4, V5（或带 _mV 后缀）
    lepod_ch_names = ["V6", "I", "II", "V1", "V2", "V3", "V4", "V5"]
    lepod_ch_mv = [f"{ch}_mV" for ch in lepod_ch_names]

    # 优先精确匹配，回退到列名包含匹配
    found_cols = []
    for candidates in [lepod_ch_names, lepod_ch_mv]:
        if all(c in df.columns for c in candidates):
            found_cols = candidates
            break

    if not found_cols:
        # 最后手段：取最后 8 列（去掉索引/时间列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 8:
            raise ValueError(
                f"CSV '{csv_path}' 中找不到 8 个数值通道。"
                f"现有列: {list(df.columns)}"
            )
        found_cols = numeric_cols[-8:]
        print(f"[lepod2mat] 警告：未找到标准列名，使用最后 8 列: {found_cols}")

    signal = df[found_cols].values.astype(np.float32)  # (N, 8)
    return signal, found_cols


def reconstruct_12_lead(signal_8ch: np.ndarray) -> np.ndarray:
    """将 Lepod 8 通道信号重建为标准 12 导联。

    输入通道顺序（按 lepod_ecg.py）: V6, I, II, V1, V2, V3, V4, V5
    输出通道顺序（PTB-XL/fairseq-signals）:
        I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    """
    # 解包 8 通道
    V6 = signal_8ch[:, 0]
    I  = signal_8ch[:, 1]
    II = signal_8ch[:, 2]
    V1 = signal_8ch[:, 3]
    V2 = signal_8ch[:, 4]
    V3 = signal_8ch[:, 5]
    V4 = signal_8ch[:, 6]
    V5 = signal_8ch[:, 7]

    # Einthoven 关系（Goldberger 导联）
    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - II / 2.0
    aVF = II - I / 2.0

    # 拼成 (N, 12)，然后转置为 (12, N)
    leads = np.stack([I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6], axis=1)
    return leads.T  # (12, N)


def resample_to_500hz(signal_12ch: np.ndarray, src_rate: int) -> np.ndarray:
    """将 (12, N) 信号从 src_rate 上采样到 500Hz。"""
    if src_rate == TARGET_SAMPLE_RATE:
        return signal_12ch

    up = TARGET_SAMPLE_RATE
    down = src_rate
    # 最简整数比：如 250Hz → 500Hz 为 2:1
    from math import gcd
    g = gcd(up, down)
    up //= g
    down //= g

    resampled = scipy.signal.resample_poly(signal_12ch, up, down, axis=1)
    return resampled.astype(np.float32)


def trim_or_pad(signal_12ch: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    """裁剪或补零到 target_len 个采样点（12, target_len）。"""
    n = signal_12ch.shape[1]
    if n >= target_len:
        return signal_12ch[:, :target_len]
    # 补零
    pad = np.zeros((12, target_len - n), dtype=np.float32)
    return np.concatenate([signal_12ch, pad], axis=1)


def convert(csv_path: str, output_path: str = None, src_rate: int = 250,
            full: bool = False) -> str:
    """主转换函数。返回输出 .mat 路径。

    默认裁/补到 10s（5000 采样）给单窗诊断用；full=True 保留**整段连续**信号（feats (12, N)），
    供 NPU 门控滑窗（npu_gate/ecg_gate_npu.py --mat）跑完整录制。两种都写同样的 schema
    （feats + curr_sample_rate），下游工具/门控通用。"""
    if output_path is None:
        base = os.path.splitext(csv_path)[0]
        output_path = base + (".mat" if not full else "_full.mat")

    print(f"[lepod2mat] 读取: {csv_path}")
    signal_8ch, cols = load_lepod_csv(csv_path)
    print(f"  原始信号: {signal_8ch.shape} @ {src_rate}Hz，通道: {cols}")

    signal_12ch = reconstruct_12_lead(signal_8ch)
    print(f"  12导联重建: {signal_12ch.shape}")

    signal_500 = resample_to_500hz(signal_12ch, src_rate)
    print(f"  上采样到 {TARGET_SAMPLE_RATE}Hz: {signal_500.shape}")

    if full:
        signal_final = signal_500
        print(f"  保留整段（--full）: {signal_final.shape} = {signal_final.shape[1]/TARGET_SAMPLE_RATE:.1f}s")
    else:
        signal_final = trim_or_pad(signal_500, TARGET_SAMPLES)
        print(f"  截断/填充到 {TARGET_SAMPLES} 个采样点: {signal_final.shape}")

    mat_data = {
        "feats": signal_final,            # (12, N) float32，mV（默认 N=5000；--full 为整段）
        "curr_sample_rate": np.array([[TARGET_SAMPLE_RATE]]),
    }
    scipy.io.savemat(output_path, mat_data)
    print(f"[lepod2mat] 已写入: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Lepod CSV → ECG-Agent .mat 转换器")
    parser.add_argument("csv", help="输入 CSV 文件路径（lepod_ecg.py 输出）")
    parser.add_argument("-o", "--output", default=None, help="输出 .mat 路径（默认同名）")
    parser.add_argument("--rate", type=int, default=250, help="Lepod 采样率（默认 250Hz）")
    parser.add_argument("--full", action="store_true",
                        help="保留整段连续信号（不截断到 10s），供门控滑窗跑完整录制")
    args = parser.parse_args()

    out = convert(args.csv, args.output, args.rate, full=args.full)
    print(f"完成: {out}")


if __name__ == "__main__":
    main()
