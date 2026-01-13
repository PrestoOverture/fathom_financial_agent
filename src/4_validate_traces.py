from pathlib import Path
from typing import Tuple
import json
import pandas as pd
import re
import random
from textwrap import fill

ROOT_DIR = Path(__file__).resolve().parents[1]

# input file paths
FINE_TUNE_JSONL_PATH = ROOT_DIR / "data" / "train" / "financebench_traces_train.jsonl"
TRAIN_CSV_PATH = ROOT_DIR / "data" / "train" / "financebench_train.csv"
TEST_JSONL_PATH = ROOT_DIR / "data" / "test" / "test.jsonl"

# output file paths
CLEANED_PATH = ROOT_DIR / "data" / "train" / "financebench_traces_train_cleaned.jsonl"
REJECTED_PATH = ROOT_DIR / "data" / "train" / "financebench_traces_train_rejected.jsonl"
REPORT_PATH = ROOT_DIR / "data" / "train" / "financebench_traces_train_report.json"

MIN_REASONING_LENGTH = 50
REQUIRED_KEYS = {"instruction", "input", "output"}

def get_holdout_ids(filepath):
    ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ids.add(record["financebench_id"])
    return ids

def validate_structure(trace):
    for key in REQUIRED_KEYS:
        if key not in trace:
            return False, f"Missing required key: {key}"
    output = trace["output"]

    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", output, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)

    if not reasoning_match:
        return False, "Missing reasoning"
    if not answer_match:
        return False, "Missing answer"
    
    reasoning_content = reasoning_match.group(1).strip()
    answer_content = answer_match.group(1).strip()

    if len(reasoning_content) < MIN_REASONING_LENGTH:
        return False, "Reasoning too short"
    if not answer_content:
        return False, "Answer is empty"
    
    return True, "ok"

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[$,%]", "", text) 
    text = text.replace(" billion", "b").replace(" million", "m")
    return text.strip()

def check_alignment(predicted_raw: str, gold_raw: str) -> bool:
    match = re.search(r"<answer>(.*?)</answer>", predicted_raw, re.DOTALL)
    if not match:
        return False
    
    pred_val = normalize_text(match.group(1))
    gold_val = normalize_text(gold_raw)

    if pred_val == gold_val:
        return True
    
    if gold_val in pred_val or pred_val in gold_val:
        return True
    
    p_num = float(re.findall(r"[-+]?\d*\.\d+|\d+", pred_val)[0])
    g_num = float(re.findall(r"[-+]?\d*\.\d+|\d+", gold_val)[0])
    if abs(p_num - g_num) / (abs(g_num) + 1e-9) < 0.05: # 5% tolerance
        return True

    return False


def main():
    print(f"Validating traces...")
    print(f"Loading traces from {FINE_TUNE_JSONL_PATH}...")

    traces = []
    with open(FINE_TUNE_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))
    
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df["financebench_id"] = train_df["financebench_id"].astype(str)
    correct_answer_lookup = train_df.set_index("financebench_id")["answer"].to_dict()
    print(correct_answer_lookup["financebench_id_01487"]) # check
    holdout_ids = get_holdout_ids(TEST_JSONL_PATH)
    print(holdout_ids)
    cleaned, rejected = [], []
    stats = {"total": len(traces), "cleaned": 0, "rejected": 0, "reasons": {}}
    print("-" * 100)
    print(traces[0])

    for t in traces:
        fb_id = t.get("metadata", {}).get("financebench_id")

        # missing id check
        if not fb_id:
            reason = "Missing financebench_id"
            t["rejected_reason"] = reason
            rejected.append(t)
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        
        fb_id = str(fb_id)

        # structure check
        is_valid, reason = validate_structure(t)
        if not is_valid:
            t["rejected_reason"] = reason
            rejected.append(t)
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue

        # leakage check
        if fb_id in holdout_ids:
            reason = "Leakage"
            t["rejected_reason"] = reason
            rejected.append(t)
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue

        # misalignment check
        if fb_id not in correct_answer_lookup:
            reason = "Misalignment"
            t["rejected_reason"] = reason
            rejected.append(t)
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue
        
        correct_answer = correct_answer_lookup[fb_id]
        predicted_output = t["output"]
        is_aligned = check_alignment(predicted_output, correct_answer)

        if not is_aligned:
            reason = "Answer Mismatch"
            t["rejected_reason"] = reason
            t["debug_gold"] = correct_answer
            match = re.search(r"<answer>(.*?)</answer>", predicted_output, re.DOTALL)
            t["debug_pred"] = match.group(1) if match else "ERR"
            
            rejected.append(t)
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue
        
        cleaned.append(t)
        stats["cleaned"] += 1
    
    print(f"Cleaned {stats['cleaned']} traces out of {stats['total']}")
    with open(CLEANED_PATH, "w") as f:
        for t in cleaned:
            f.write(json.dumps(t) + "\n")
    print(f"Rejected {stats['rejected']} traces out of {stats['total']}")
    with open(REJECTED_PATH, "w") as f:
        for t in rejected:
            f.write(json.dumps(t) + "\n")
    with open(REPORT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\nValidation Summary:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
    with open(CLEANED_PATH, "r") as f:
        traces = [json.loads(line) for line in f]

    samples = random.sample(traces, 3)

    for i, s in enumerate(samples, 1):
        print(f"\n{'='*20} SAMPLE {i} {'='*20}")
        print(f"QUESTION: {s['input'].split('Question:')[-1].strip()}")
        print("-" * 10 + " OUTPUT " + "-" * 10)
        print(fill(s['output'], width=100))
        print(f"\nMETADATA: {s.get('metadata', {}).get('financebench_id')}")