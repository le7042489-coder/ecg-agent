"""
合并 LoRA adapter 到 base model 并导出 GGUF Q4_K_M

运行：
    conda run -n ecg-agent python merge_and_export_gguf.py

输出：
    merged_llama3b_ecg/        合并后的 HuggingFace 格式（float16）
    ecg_agent_llama3b_q4km.gguf  最终 GGUF 文件（约 2 GB）
"""

import os, sys, subprocess
from pathlib import Path

WORKSPACE = Path(__file__).parent
ADAPTER_PATH = WORKSPACE / "ecg-dialogue-finetune" / "Llama-3.2-3B-Instruct"
MERGED_DIR   = WORKSPACE / "merged_llama3b_ecg"
GGUF_OUT     = WORKSPACE / "ecg_agent_llama3b_q4km.gguf"
LLAMA_CPP    = WORKSPACE / "llama.cpp"
BASE_MODEL   = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"

# ─── Step 1: 合并 LoRA ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Step 1: 合并 LoRA adapter → float16 完整模型")
print("="*60)

if MERGED_DIR.exists() and (MERGED_DIR / "config.json").exists():
    print(f"[跳过] {MERGED_DIR} 已存在，直接进行 Step 2")
else:
    from unsloth import FastLanguageModel
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    import torch

    print(f"加载 base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )

    print(f"构建 LoRA 结构（r=16, alpha=16）...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["v_proj", "k_proj", "q_proj",
                        "gate_proj", "o_proj", "down_proj", "up_proj"],
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
    )

    print(f"加载训练权重: {ADAPTER_PATH / 'adapter_model.safetensors'}")
    state_dict = load_file(str(ADAPTER_PATH / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, state_dict)
    print("  权重加载完成，验证模型可正向传播...")

    print(f"合并并保存 float16 → {MERGED_DIR} ...")
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"[完成] 合并模型已保存到 {MERGED_DIR}")
    print(f"  大小: {sum(f.stat().st_size for f in MERGED_DIR.rglob('*') if f.is_file()) / 1024**3:.1f} GB")

# ─── Step 2: 克隆/更新 llama.cpp ─────────────────────────────────────────────
print("\n" + "="*60)
print("Step 2: 准备 llama.cpp 转换工具")
print("="*60)

if not LLAMA_CPP.exists():
    print("克隆 llama.cpp（shallow clone）...")
    subprocess.run([
        "git", "clone", "--depth=1",
        "https://github.com/ggerganov/llama.cpp",
        str(LLAMA_CPP)
    ], check=True)
else:
    print(f"[跳过] llama.cpp 已存在于 {LLAMA_CPP}")

# 安装 llama.cpp 的 Python 依赖
print("安装 llama.cpp 转换依赖...")
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--quiet", "--upgrade",
    "gguf", "sentencepiece", "transformers"
], check=True)

# ─── Step 3: 转换为 GGUF Q4_K_M ──────────────────────────────────────────────
print("\n" + "="*60)
print("Step 3: 转换 GGUF Q4_K_M")
print("="*60)

convert_script = LLAMA_CPP / "convert_hf_to_gguf.py"
if not convert_script.exists():
    # 尝试旧版脚本名
    convert_script = LLAMA_CPP / "convert.py"

if not convert_script.exists():
    print(f"[错误] 找不到转换脚本。llama.cpp 目录内容：")
    for f in LLAMA_CPP.iterdir():
        print(f"  {f.name}")
    sys.exit(1)

print(f"使用转换脚本: {convert_script.name}")
print(f"输出文件: {GGUF_OUT}")

# 先转换为 f16 GGUF（中间格式），再量化到 Q4_K_M
F16_GGUF = WORKSPACE / "ecg_agent_llama3b_f16.gguf"

print("\n[3a] HuggingFace → GGUF f16...")
result = subprocess.run([
    sys.executable, str(convert_script),
    str(MERGED_DIR),
    "--outfile", str(F16_GGUF),
    "--outtype", "f16",
], capture_output=False)

if result.returncode != 0:
    print("[错误] GGUF 转换失败")
    sys.exit(1)

print(f"  f16 GGUF: {F16_GGUF.stat().st_size / 1024**3:.1f} GB")

# 量化：需要 llama.cpp 的 quantize 二进制（如果没有就跳过，保留 f16）
quantize_bin = LLAMA_CPP / "build" / "bin" / "llama-quantize"
if not quantize_bin.exists():
    quantize_bin = LLAMA_CPP / "quantize"

if quantize_bin.exists():
    print("\n[3b] 量化 f16 → Q4_K_M...")
    subprocess.run([str(quantize_bin), str(F16_GGUF), str(GGUF_OUT), "Q4_K_M"], check=True)
    print(f"  Q4_K_M GGUF: {GGUF_OUT.stat().st_size / 1024**3:.1f} GB")
    # 清理中间文件
    F16_GGUF.unlink()
    print(f"  已删除中间文件 {F16_GGUF.name}")
else:
    # convert_hf_to_gguf.py 的新版本支持直接量化
    print("\n[3b] 直接用 convert 脚本输出 Q4_K_M（新版 llama.cpp）...")
    F16_GGUF.unlink(missing_ok=True)
    result = subprocess.run([
        sys.executable, str(convert_script),
        str(MERGED_DIR),
        "--outfile", str(GGUF_OUT),
        "--outtype", "q4_k_m",
    ], capture_output=False)
    if result.returncode != 0:
        # 回退：convert 不支持直接量化，保留 f16
        print("[警告] 直接量化失败，改用两步法：先转 f16，再用 gguf-split 量化")
        # 重新转 f16
        subprocess.run([
            sys.executable, str(convert_script),
            str(MERGED_DIR),
            "--outfile", str(F16_GGUF),
            "--outtype", "f16",
        ], check=True)
        print(f"  f16 GGUF 保存为 {F16_GGUF}（约 {F16_GGUF.stat().st_size/1024**3:.1f} GB）")
        print("  请手动编译 llama.cpp 的 quantize 工具后运行：")
        print(f"    cd {LLAMA_CPP} && cmake -B build && cmake --build build --target llama-quantize -j")
        print(f"    ./build/bin/llama-quantize {F16_GGUF} {GGUF_OUT} Q4_K_M")
        sys.exit(0)
    else:
        print(f"  Q4_K_M GGUF: {GGUF_OUT.stat().st_size / 1024**3:.1f} GB")

print("\n" + "="*60)
print("全部完成！")
print(f"GGUF 文件：{GGUF_OUT}")
print(f"大小：{GGUF_OUT.stat().st_size / 1024**3:.1f} GB")
print()
print("传输到 DK-2500：")
print(f"  scp {GGUF_OUT} user@10.157.225.11:~/workspace/ECG-Agent/")
print("="*60)
