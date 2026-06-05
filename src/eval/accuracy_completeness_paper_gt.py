#!/usr/bin/env python3
"""
Accuracy & Completeness evaluation with PAPER-FAITHFUL ground truth.

Difference from accuracy_completeness.py:
  - accuracy_completeness.py scores the model response against the ECG-MTD
    dataset's own synthetic assistant reply (`ground_truth_dialogue`).
  - THIS script reconstructs the paper's ground truth (ECG-Agent.pdf, Sec 3.1/3.3):
      * post_classification queries -> ground truth generated from the
        cardiologist-labeled diagnostic codes in PTB-XL (scp_codes + scp_statements).
      * post_measurement queries   -> ground truth generated from the
        University of Glasgow (Uni-G) PQRST interval measurements in PTB-XL+.
    For each query we first prompt the judge LLM to write the ideal ground-truth
    answer FROM the reference data, then we score the model's answer against that
    generated ground truth (same 1-5 Accuracy / Completeness rubric as the paper).
  - direct_response turns have no clinical reference -> we keep the ECG-MTD reply
    as ground truth (the paper had no reference data for non-tool queries either).

Judge LLM: DeepSeek (set DEEPSEEK_API_KEY). The paper used Gemini-2.5-Pro; the
judge model is intentionally a free choice here (per user request).

Two-stage, both cached/resumable:
  Stage A  build reference-grounded GT answer per (ecg_id, category, query)
  Stage B  score model gen_content vs that GT answer
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = ["post_classification", "post_measurement", "direct_response"]


# --------------------------------------------------------------------------- #
# Reference-data loading
# --------------------------------------------------------------------------- #
def load_scp_descriptions(scp_statements_path):
    """code -> human-readable description (+ diagnostic class)."""
    import csv

    desc = {}
    with open(scp_statements_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # first column is the code (unnamed), 'description' is col index 1
        for row in reader:
            if not row:
                continue
            code = row[0].strip()
            description = row[1].strip() if len(row) > 1 else ""
            desc[code] = description
    return desc


def load_ptbxl_codes(ptbxl_db_path):
    """ecg_id(int) -> dict(code -> likelihood float)."""
    import csv

    out = {}
    with open(ptbxl_db_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ecg_id = int(row["ecg_id"])
                codes = eval(row["scp_codes"])  # PTB-XL stores a python dict literal
                out[ecg_id] = codes
            except Exception:
                continue
    return out


# Uni-G global columns -> (label, unit). QTc uses Bazett to match the agent tool.
UNIG_COLS = [
    ("HR__Global", "Heart rate", "bpm"),
    ("HR_Ventr_Global", "Heart rate", "bpm"),  # fallback for HR
    ("PR_Int_Global", "PR interval", "ms"),
    ("QRS_Dur_Global", "QRS duration", "ms"),
    ("QT_Int_Global", "QT interval", "ms"),
    ("QT_IntBazett_Global", "QTc (Bazett)", "ms"),
    ("QT_IntCorr_Global", "QTc (Bazett)", "ms"),  # fallback for QTc
]


def load_unig_measurements(unig_path):
    """ecg_id(int) -> ordered dict of {label: 'value unit'} using global Uni-G cols."""
    import csv

    wanted = {c for c, _, _ in UNIG_COLS}
    out = {}
    with open(unig_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ecg_id = int(float(row["ecg_id"]))
            except Exception:
                continue
            vals = {}
            for col, label, unit in UNIG_COLS:
                if label in vals:  # already filled by a primary column
                    continue
                raw = row.get(col, "")
                if raw is None or str(raw).strip() in ("", "nan", "NaN"):
                    continue
                try:
                    v = float(raw)
                except Exception:
                    continue
                vals[label] = f"{v:.1f} {unit}".strip()
            if vals:
                out[ecg_id] = vals
    return out


# --------------------------------------------------------------------------- #
# Reference text builders
# --------------------------------------------------------------------------- #
def classification_reference_text(ecg_id, codes_map, desc_map):
    codes = codes_map.get(ecg_id)
    if not codes:
        return None
    parts = []
    for code, likelihood in codes.items():
        d = desc_map.get(code, code)
        if likelihood and float(likelihood) > 0:
            parts.append(f"{code} ({d}) [likelihood {float(likelihood):.0f}%]")
        else:
            parts.append(f"{code} ({d})")
    return "; ".join(parts)


def measurement_reference_text(ecg_id, unig_map):
    vals = unig_map.get(ecg_id)
    if not vals:
        return None
    return "; ".join(f"{label}: {v}" for label, v in vals.items())


# --------------------------------------------------------------------------- #
# DeepSeek client
# --------------------------------------------------------------------------- #
def make_client(api_key):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def gt_generation_prompt(category, user_query, reference_text):
    if category == "post_classification":
        source = ("authoritative cardiologist-confirmed diagnostic codes from the "
                  "PTB-XL database (the SCP-ECG statements assigned by the labeling "
                  "cardiologists for this 12-lead ECG)")
    else:
        source = ("authoritative ECG interval measurements computed by the University "
                  "of Glasgow (Uni-G) ECG analysis program, as distributed in PTB-XL+")
    return f"""You are a cardiologist's assistant writing the ideal reference answer for a patient.

The patient asks: "{user_query}"

The following are the {source}:
{reference_text}

Write the single best ground-truth answer to the patient's question, accurately and
faithfully conveying ONLY what these reference findings support. Be clinically correct,
concise, and patient-friendly. Do not invent findings beyond the reference data. Output
only the answer text."""


def score_prompt(category, ecg_id, user_query, tool_output, gt_response, model_response):
    cat_label = category
    return f"""
You are a medical expert evaluating an AI's ECG dialogue response. Compare the assistant's response to the ground truth.

**Context:**
- ECG ID: {ecg_id}
- Response Category: {cat_label}
- Tool Output (provided to assistant): {tool_output if tool_output is not None else 'N/A'}
- User's Question: {user_query}

**Ground Truth Response:**
{gt_response}

**Assistant Response to Evaluate:**
{model_response}

**Evaluation Criteria:**
1. Accuracy (1-5): How well does the response match the ground truth? Key information to look for are representative diagnosis classes (e.g., sinus rhythm, myocardial infarction) or measurement interval (e.g., Heart rate, PR interval, QRS duration, QTc interval) for post tool calls and representative key information for direct responses.
    - 5: Fully accurate (all representative diagnosis classes identified and correct; measurements accurate).
    - 3: Partially accurate (about half correct).
    - 1: Largely inaccurate (no relevant class identified; measurements wrong).
2. Completeness (1-5): Does the response cover essential information mentioned in the ground truth?
    - 5: Comprehensive (includes all key information from the ground truth).
    - 3: Partially complete (includes some, misses others).
    - 1: Incomplete (missing all key information).

**Instructions:** Rate each criterion and provide a brief justification.

**Response Format:**
Accuracy: [score]
Accuracy Explanation: [justification]
Completeness: [score]
Completeness Explanation: [justification]
"""


def parse_scores(text):
    out = {}
    for crit in ("accuracy", "completeness"):
        m = re.search(rf"^{crit}:\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            out[crit] = int(m.group(1))
    return out


# --------------------------------------------------------------------------- #
# JSONL -> evaluation items
# --------------------------------------------------------------------------- #
def ecg_id_from_file(ecg_file):
    if isinstance(ecg_file, str) and ecg_file.strip().startswith("["):
        try:
            ecg_file = json.loads(ecg_file)[0]
        except Exception:
            pass
    if isinstance(ecg_file, list):
        ecg_file = ecg_file[0] if ecg_file else ""
    m = re.search(r"(\d+)", ecg_file or "")
    return int(m.group(1)) if m else None


def collect_items(inference_file):
    """Yield evaluation items from `turns`, aligned model response vs category."""
    items = []
    with open(inference_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ecg_id = ecg_id_from_file(rec.get("ecg_file", ""))
            if ecg_id is None:
                continue
            turns = rec.get("turns", [])
            for i, t in enumerate(turns):
                if t.get("gt_action") != "response":
                    continue
                prev = turns[i - 1] if i > 0 else {}
                pa = prev.get("gt_action")
                if pa == "call_classification_tool":
                    category = "post_classification"
                elif pa == "call_measurement_tool":
                    category = "post_measurement"
                elif prev.get("gt_action") == "response_followup" or i == 0 or prev.get("user_content"):
                    category = "direct_response"
                else:
                    category = "direct_response"
                # model's response at this turn
                model_response = t.get("gen_content") if t.get("gen_action") == "response" else t.get("gen_content")
                items.append({
                    "ecg_id": ecg_id,
                    "category": category,
                    "user_query": t.get("user_content") or "",
                    "tool_output": prev.get("gt_tool_output") if category != "direct_response" else None,
                    "model_response": (model_response or "").strip(),
                    "dataset_gt": (t.get("gt_content") or "").strip(),
                    "turn_index": t.get("turn_index"),
                })
    return items


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_file", required=True)
    ap.add_argument("--output_dir", default="results/accuracy_completeness_paper_gt")
    ap.add_argument("--ptbxl_db", default="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")
    ap.add_argument("--scp_statements", default="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv")
    ap.add_argument("--unig_features", default="data/ptbxl_plus/unig_features.csv")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--max_per_category", type=int, default=None,
                    help="Cap items per category (debug). Default: all.")
    ap.add_argument("--workers", type=int, default=12,
                    help="Concurrent API requests. Default: 12.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not set.")
        sys.exit(1)

    logger.info("Loading reference data...")
    desc_map = load_scp_descriptions(args.scp_statements)
    codes_map = load_ptbxl_codes(args.ptbxl_db)
    unig_map = load_unig_measurements(args.unig_features)
    logger.info(f"PTB-XL codes: {len(codes_map)} ecg_ids | Uni-G measurements: {len(unig_map)} ecg_ids "
                f"| scp descriptions: {len(desc_map)}")

    items = collect_items(args.inference_file)
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    logger.info("Items by category: " + ", ".join(f"{k}={len(v)}" for k, v in by_cat.items()))

    if args.max_per_category:
        items = []
        for c in CATEGORIES:
            items.extend(by_cat[c][: args.max_per_category])
    # reference-data coverage check
    miss_cls = [it for it in items if it["category"] == "post_classification" and it["ecg_id"] not in codes_map]
    miss_mea = [it for it in items if it["category"] == "post_measurement" and it["ecg_id"] not in unig_map]
    if miss_cls:
        logger.warning(f"{len(miss_cls)} classification items missing PTB-XL codes (will be skipped).")
    if miss_mea:
        logger.warning(f"{len(miss_mea)} measurement items missing Uni-G measurements (will be skipped).")

    client = make_client(api_key)

    gt_cache_path = os.path.join(args.output_dir, "reference_gt_cache.json")
    results_path = os.path.join(args.output_dir, "scored_items.jsonl")
    gt_cache = {}
    if os.path.exists(gt_cache_path):
        gt_cache = json.load(open(gt_cache_path))

    # resume: which (ecg_id, turn_index) already scored
    done = set()
    if os.path.exists(results_path):
        for line in open(results_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["ecg_id"], r["turn_index"]))

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_lock = threading.Lock()
    write_lock = threading.Lock()
    out_f = open(results_path, "a")

    todo = [it for it in items if (it["ecg_id"], it["turn_index"]) not in done]
    n_total = len(todo)
    logger.info(f"{len(items)} items, {len(done)} already done, {n_total} to process "
                f"with {args.workers} workers.")
    progress = {"n": 0}

    def process(it):
        cat = it["category"]
        ecg_id = it["ecg_id"]

        # --- Stage A: reference-grounded GT ---
        if cat == "post_classification":
            ref = classification_reference_text(ecg_id, codes_map, desc_map)
            if not ref:
                return None
        elif cat == "post_measurement":
            ref = measurement_reference_text(ecg_id, unig_map)
            if not ref:
                return None
        else:
            ref = None

        if cat == "direct_response":
            gt_answer = it["dataset_gt"]  # no clinical reference -> dataset GT
            gt_source = "dataset"
        else:
            ck = f"{cat}|{ecg_id}|{it['turn_index']}"
            with cache_lock:
                gt_answer = gt_cache.get(ck)
            if gt_answer is None:
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user",
                                   "content": gt_generation_prompt(cat, it["user_query"], ref)}],
                        max_tokens=4096,  # deepseek reasoning tokens count toward budget
                    )
                    gt_answer = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    logger.error(f"GT-gen failed ecg {ecg_id} turn {it['turn_index']}: {e}")
                    return None
                if not gt_answer:
                    logger.warning(f"Empty GT for ecg {ecg_id} turn {it['turn_index']} "
                                   f"(finish={resp.choices[0].finish_reason}); skipping (not cached).")
                    return None
                with cache_lock:
                    gt_cache[ck] = gt_answer
                    json.dump(gt_cache, open(gt_cache_path, "w"), ensure_ascii=False, indent=1)
            gt_source = "reference"

        if not gt_answer:
            return None

        # --- Stage B: score model response vs GT ---
        if not it["model_response"]:
            score = {}
            err = "Model produced no response at this turn."
        else:
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user",
                               "content": score_prompt(cat, ecg_id, it["user_query"],
                                                       it["tool_output"], gt_answer,
                                                       it["model_response"])}],
                    max_tokens=4096,  # leave room for deepseek reasoning + structured answer
                )
                txt = resp.choices[0].message.content or ""
                score = parse_scores(txt)
                err = None if score else "parse_failed"
            except Exception as e:
                score = {}
                err = str(e)

        return {
            "ecg_id": ecg_id, "turn_index": it["turn_index"], "category": cat,
            "user_query": it["user_query"], "reference_data": ref,
            "gt_source": gt_source, "ground_truth_response": gt_answer,
            "model_response": it["model_response"],
            "accuracy": score.get("accuracy"), "completeness": score.get("completeness"),
            "error": err,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process, it) for it in todo]
        for fut in as_completed(futures):
            rec = fut.result()
            progress["n"] += 1
            if rec is None:
                continue
            with write_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
            if progress["n"] % 10 == 0 or progress["n"] == n_total:
                logger.info(f"[{progress['n']}/{n_total}] {rec['category']} ecg {rec['ecg_id']} "
                            f"-> acc={rec['accuracy']} comp={rec['completeness']}")
    out_f.close()

    # --- aggregate ---
    agg = {c: {"accuracy": [], "completeness": [], "n": 0} for c in CATEGORIES}
    for line in open(results_path):
        if not line.strip():
            continue
        r = json.loads(line)
        c = r["category"]
        if r.get("accuracy") is not None and r.get("completeness") is not None:
            agg[c]["accuracy"].append(r["accuracy"])
            agg[c]["completeness"].append(r["completeness"])
            agg[c]["n"] += 1

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    report = ["# Accuracy & Completeness — Paper-Faithful Ground Truth",
              f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              f"Inference file: `{args.inference_file}`",
              f"Judge LLM: {args.model}",
              "",
              "GT sources: post_classification ← PTB-XL cardiologist scp_codes; "
              "post_measurement ← PTB-XL+ Uni-G PQRST intervals; "
              "direct_response ← ECG-MTD dataset reply (no clinical reference).",
              "",
              "| Category | Valid | Avg Accuracy | Avg Completeness |",
              "|----------|-------|--------------|------------------|"]
    label = {"post_classification": "Post-Classification",
             "post_measurement": "Post-Measurement",
             "direct_response": "Direct Response"}
    for c in CATEGORIES:
        a = agg[c]
        report.append(f"| {label[c]} | {a['n']} | {avg(a['accuracy']):.2f} | {avg(a['completeness']):.2f} |")
    report_txt = "\n".join(report)
    with open(os.path.join(args.output_dir, "evaluation_report.md"), "w") as f:
        f.write(report_txt + "\n")
    print("\n" + report_txt + "\n")
    logger.info(f"Done. Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
