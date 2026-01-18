from pathlib import Path
from typing import List, Dict
import json
import pandas as pd
import random
from transformers import AutoTokenizer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_TRACES_PATH = ROOT_DIR / "data" / "train" / "financebench_traces_train_cleaned.jsonl"
INPUT_CSV_PATH = ROOT_DIR / "data" / "train" / "financebench_train.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "train"
MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
MAX_TOKENS = 4096
SEED = 3407

def load_traces(path: Path) -> List[Dict]:
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))
    return traces

def save_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def trim_context(text, tokenizer, max_len=4096, overhead=500):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= (max_len - overhead):
        return text
    
    target_len = max_len - overhead
    
    keep_start = int(target_len * 0.2)
    keep_end = int(target_len * 0.8)
    
    start_text = tokenizer.decode(tokens[:keep_start])
    end_text = tokenizer.decode(tokens[-keep_end:])
    
    return f"{start_text}\n\n[...Evidence Trimmed for Length...]\n\n{end_text}"

def trim_set(split_name, traces, tokenizer):
    final = []
    dropped = 0
    trimmed = 0

    for trace in tqdm(traces, desc=f"Trimming {split_name}"):
        system_prompt = trace.get("instruction", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": trace["input"]},
            {"role": "assistant", "content": trace["output"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        num_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

        if num_tokens > MAX_TOKENS:
            trace["input"] = trim_context(trace["input"], tokenizer, MAX_TOKENS)
            trimmed += 1

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": trace["input"]},
                {"role": "assistant", "content": trace["output"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            num_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

            if num_tokens > MAX_TOKENS:
                dropped += 1
                continue

        final.append(trace)

    print(f"Processed {split_name}: {len(final)} kept, {trimmed} trimmed, {dropped} dropped.")
    return final, trimmed, dropped


def main():
    traces = load_traces(INPUT_TRACES_PATH)
    print(f"Loaded {len(traces)} traces")

    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_CSV_PATH}")
    id_to_doc = dict(zip(df['financebench_id'], df['doc_name']))
    doc_groups = {}
    print(id_to_doc)
    for trace in traces:
        financebench_id = trace.get('metadata', {}).get('financebench_id')
        # print(financebench_id)
        doc_name = id_to_doc.get(financebench_id)
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(trace)

    print(f"Found {len(doc_groups)} unique documents across {len(traces)} traces.")
    print(doc_groups.keys())

    random.seed(SEED)
    all_docs = list(doc_groups.keys())
    print(f"Total documents: {len(all_docs)}")
    random.shuffle(all_docs)
    print(all_docs[:5])

    probe_docs = all_docs[:5]
    remaining_docs = all_docs[5:]

    num_dev = max(1, int(len(remaining_docs) * 0.1))
    print(f"Number of development documents: {num_dev}")
    dev_docs = remaining_docs[:num_dev]
    train_docs = remaining_docs[num_dev:]
    print(f"Split: Probe={len(probe_docs)} docs, Dev={len(dev_docs)} docs, Train={len(train_docs)} docs")

    probe_set = [t for d in probe_docs for t in doc_groups[d]]
    dev_set = [t for d in dev_docs for t in doc_groups[d]]
    train_set = [t for d in train_docs for t in doc_groups[d]]

    print(f"Probe set size: {len(probe_set)}")
    print(f"Dev set size: {len(dev_set)}")
    print(f"Train set size: {len(train_set)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    probe_set, probe_trimmed, probe_dropped = trim_set("Probe", probe_set, tokenizer)
    dev_set, dev_trimmed, dev_dropped = trim_set("Dev", dev_set, tokenizer)
    final_train, train_trimmed, train_dropped = trim_set("Train", train_set, tokenizer)

    save_jsonl(probe_set, f"{OUTPUT_DIR}/probe.jsonl")
    save_jsonl(dev_set, f"{OUTPUT_DIR}/dev.jsonl")
    save_jsonl(final_train, f"{OUTPUT_DIR}/train.jsonl")
    
    splits = {
        "probe_docs": probe_docs,
        "dev_docs": dev_docs,
        "train_docs": train_docs,
        "stats": {
            "probe_count": len(probe_set),
            "dev_count": len(dev_set),
            "train_count": len(final_train),
            "probe_trimmed": probe_trimmed,
            "dev_trimmed": dev_trimmed,
            "train_trimmed": train_trimmed,
            "probe_dropped": probe_dropped,
            "dev_dropped": dev_dropped,
            "train_dropped": train_dropped,
        }
    }
    with open(f"{OUTPUT_DIR}/splits.json", 'w') as f:
        json.dump(splits, f, indent=2)
    
if __name__ == "__main__":
    main()