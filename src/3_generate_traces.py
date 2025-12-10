import argparse
import json
import os
from pathlib import Path
from typing import Tuple
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pandas import DataFrame

"""
uv run python src/3_generate_traces.py split
uv run python src/3_generate_traces.py generate-batch-input
uv run python src/3_generate_traces.py chunk-batch-input --max-batch-tokens <max_batch_tokens>
uv run python src/3_generate_traces.py submit-batch --batch-file <batch_input_file_name>       # note the printed batch id
uv run python src/3_generate_traces.py debug-batch --batch-id <batch_id_from_previous_step> # check progress and errors
uv run python src/3_generate_traces.py download-batch-results --batch-id <batch_id_from_previous_step> # download the results
uv run python src/3_generate_traces.py convert-output
"""

ROOT_DIR = Path(__file__).resolve().parents[1] # all file paths remain absolute and stable, independent of where the script is run from
INPUT_FILE_PATH = ROOT_DIR / "data" / "audited" / "financebench_filtered_reasoning.csv"
TRAIN_CSV_PATH = ROOT_DIR / "data" / "train" / "financebench_train.csv"
TEST_CSV_PATH = ROOT_DIR / "data" / "test" / "financebench_test.csv"
TEST_JSONL_PATH = ROOT_DIR / "data" / "test" / "test.jsonl"
BATCH_INPUT_PATH = ROOT_DIR / "data" / "train" / "teacher_batch_input.jsonl"
BATCH_OUTPUT_PATH = ROOT_DIR / "data" / "train" / "teacher_batch_output.jsonl"
FINE_TUNE_JSONL_PATH = ROOT_DIR / "data" / "train" / "financebench_traces_train.jsonl"
BATCHES_INPUT_DIR = ROOT_DIR / "data" / "train" / "batches_input"
BATCHES_OUTPUT_DIR = ROOT_DIR / "data" / "train" / "batches_output"
MAX_BATCH_TOKENS = 85000

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE_PATH)
    return df

def split_train_test(df: DataFrame, n_test: int) -> Tuple[DataFrame, DataFrame]:
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = df_shuffled.iloc[:n_test].copy()
    train_df = df_shuffled.iloc[n_test:].copy()
    return train_df, test_df

def save_jsonl(test_df: DataFrame) -> None:
    cols_to_keep = ["financebench_id", "question", "answer", "evidence"]
    for col in cols_to_keep:
        if col not in test_df.columns:
            raise ValueError(f"Column {col} not found in test dataframe")
    
    TEST_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TEST_JSONL_PATH.open("w", encoding="utf-8") as f:
        for _, row in test_df.iterrows():
            record = {
                "financebench_id": str(row["financebench_id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "evidence": str(row["evidence"]),
                "metadata": {
                    "company": str(row.get("company")),
                    "doc_name": str(row.get("doc_name")),
                    "question_type": str(row.get("question_type")),
                    "question_reasoning": str(row.get("question_reasoning")),
                    "domain_question": str(row.get("domain_question")),
                    "justification": str(row.get("justification")),
                    "gics_sector": str(row.get("gics_sector")),
                    "doc_type": str(row.get("doc_type")),
                    "doc_period": str(row.get("doc_period")),
                    "doc_link": str(row.get("doc_link"))
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
def split_and_save() -> None:
    df = load_data()
    train_df, test_df = split_train_test(df, n_test=15)

    TRAIN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)
    save_jsonl(test_df)

    print(f"Train data {len(train_df)} rows saved to {TRAIN_CSV_PATH}")
    print(f"Test data {len(test_df)} rows saved to {TEST_CSV_PATH}")
    print(f"Test JSONL {len(test_df)} rows saved to {TEST_JSONL_PATH}")

def generate_batch_input(train_df: DataFrame) -> None:
    cols_to_keep = ["financebench_id", "question", "answer", "evidence"]
    for col in cols_to_keep:
        if col not in train_df.columns:
            raise ValueError(f"Column {col} not found in train dataframe")
    
    BATCH_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BATCH_INPUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            financebench_id = str(row["financebench_id"])
            question = str(row["question"])
            answer = str(row["answer"])
            evidence = str(row["evidence"])

            system_prompt = "You are an expert financial analyst."

            user_prompt = f"""
                You are given a question about a company's financial statements, the relevant evidence from the official filing, and the **correct** answer. \n
                Your job is to explain, step-by-step, how to derive that answer using ONLY the provided evidence.\n\n

                Question: {question}\n
                Evidence: {evidence}\n
                Answer: {answer}\n\n

                Task: \n
                1. Use the evidence to show how an analyst would compute or justify the correct answer.\n
                2. Write detailed, step-by-step reasoning.\n
                3. Then restate the exact correct answer.\n\n

                Output MUST follow this exact XML-like format:\n
                <reasoning>\n
                1. ...\n
                2. ...\n
                ...\n
                </reasoning>\n
                <answer>...</answer>\n\n

                Rules:\n
                - Do NOT invent numbers that are not supported by the context.\n
                - Do NOT change the final answer; it must match the given correct answer.\n
                - Make each reasoning step explicit and numerical where appropriate.
            """

            body = {
                "model": "gpt-4o",
                "temperature": 0.3,
                "max_tokens": 1000,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

            record = {
                "custom_id": f"fb-trace-{financebench_id}",
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": body
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(train_df)} traces saved to {BATCH_INPUT_PATH}")

def generate_batch_input_and_save() -> None:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    generate_batch_input(train_df)
    print(f"Batch input saved to {BATCH_INPUT_PATH}")

def estimate_request_tokens(record: dict, encoder: tiktoken.Encoding) -> int:
    """Approximate tokens per request: sum message content tokens + reserved max_tokens."""
    body = record.get("body", {})
    messages = body.get("messages", [])
    text_parts: list[str] = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(str(part["text"]))
        else:
            text_parts.append(str(content))

    input_tokens = len(encoder.encode("".join(text_parts)))
    max_tokens = int(body.get("max_tokens", 0) or 0)
    return input_tokens + max_tokens


def split_batch_file(
    batch_file_path: Path,
    output_dir: Path,
    max_batch_tokens: int = MAX_BATCH_TOKENS,
) -> list[Path]:
    """Split a JSONL batch file into multiple chunk files under the token cap."""
    if not batch_file_path.exists():
        raise FileNotFoundError(f"{batch_file_path} does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = tiktoken.get_encoding("o200k_base")

    chunks: list[Path] = []
    current_chunk: list[str] = []
    current_tokens = 0
    chunk_index = 1

    with batch_file_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            cost = estimate_request_tokens(record, encoder)

            if cost > max_batch_tokens:
                raise ValueError(
                    f"Single request {record.get('custom_id')} needs {cost} tokens, "
                    f"exceeds chunk limit {max_batch_tokens}"
                )

            if current_tokens + cost > max_batch_tokens and current_chunk:
                chunk_path = save_chunk(output_dir, chunk_index, current_chunk)
                chunks.append(chunk_path)
                chunk_index += 1
                current_chunk = []
                current_tokens = 0

            current_chunk.append(line)
            current_tokens += cost

    if current_chunk:
        chunk_path = save_chunk(output_dir, chunk_index, current_chunk)
        chunks.append(chunk_path)

    return chunks

def save_chunk(output_dir: Path, index: int, lines: list[str]) -> Path:
    chunk_path = output_dir / f"batch_chunk_{index}.jsonl"
    with chunk_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
    print(f"Saved {chunk_path} ({len(lines)} requests)")
    return chunk_path

def submit_batch(openai: OpenAI, batch_file_name: Path) -> str:
    batch_path = BATCHES_INPUT_DIR / batch_file_name

    # must upload as binary to avoid encoding issues
    with batch_path.open("rb") as f:
        uploaded = openai.files.create(
            file=f,
            purpose="batch"
        )

    batch = openai.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h", # the only option for now
        metadata={
            "job_type": "financebench_trace_generation"
        }
    )

    print(f"Batch created: {batch.id}")
    print(f"Batch status: {batch.status}")
    print(f"Batch errors: {batch.errors}")
    print(f"Batch error file id: {batch.error_file_id}")
    print(f"Batch output file id: {batch.output_file_id}")
    print(f"Batch input file id: {batch.input_file_id}")
    print(f"Batch metadata: {batch.metadata}")
    print(f"Batch created at: {batch.created_at}")
    print(f"Batch completed at: {batch.completed_at}")
    return batch.id

def debug_batch(openai: OpenAI, batch_id: str | None, batch_file: Path | None = None) -> None:
    if not batch_id:
        if not batch_file:
            raise ValueError("Provide either --batch-id or --batch-file.")
        print(f"Submitting batch from {batch_file}...")
        batch_id = submit_batch(openai, batch_file)

    batch = openai.batches.retrieve(batch_id)
    if batch.errors:
        print(f"Batch errors: {batch.errors}")
    else:
        print("No errors found")
    print(f"Batch status: {batch.status}")
    print(f"Batch id: {batch.id}")
    print(f"Batch output file id: {batch.output_file_id}")
    print(f"Batch error file id: {batch.error_file_id}")

def download_batch_results(openai: OpenAI, output_file_id: str, destination: Path | None = None, batch_id: str | None = None) -> Path:
    if destination:
        output_path = destination
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        BATCHES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"batch_output_{batch_id or output_file_id}.jsonl"
        output_path = BATCHES_OUTPUT_DIR / name

    response = openai.files.content(output_file_id)

    # use "wb" to write binary data to the file 
    # since openai.files.content() returns a binary response
    with output_path.open("wb") as f:
        f.write(response.read())
    print(f"Batch results saved to {output_path}")
    return output_path

def load_train_index() -> dict[str, dict]:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    index: dict[str, dict] = {} # nested dictionary with financebench_id as the key
    for _, row in train_df.iterrows():
        financebench_id = str(row["financebench_id"])
        index[financebench_id] = {
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "evidence": str(row["evidence"])
        }
    return index

def convert_batch_output() -> None:
    train_index = load_train_index()
    instruction = "You are a financial analyst. Answer the user's question based on the context provided. Show your reasoning steps."
    FINE_TUNE_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    output_files = sorted(BATCHES_OUTPUT_DIR.glob("*.jsonl"))
    if not output_files:
        raise ValueError(f"No batch output files found in {BATCHES_OUTPUT_DIR}")

    with FINE_TUNE_JSONL_PATH.open("w", encoding="utf-8") as f_out:
        for out_file in output_files:
            print(f"Converting {out_file}...")
            with out_file.open("r", encoding="utf-8") as f_in:
                for line in f_in:
                    record = json.loads(line)
                    if record.get("error"):
                        raise ValueError(f"Error in batch output {out_file}: {record['error']}")
                    if "response" not in record or "body" not in record["response"]:
                        raise ValueError(f"Batch output record missing response body in {out_file}")

                    custom_id = record["custom_id"]
                    financebench_id = custom_id.replace("fb-trace-", "")
                    if financebench_id not in train_index:
                        raise ValueError(f"Financebench ID {financebench_id} not found in train index")

                    row = train_index[financebench_id]
                    body = record["response"]["body"]
                    choice = body["choices"][0]
                    content = choice["message"]["content"]

                    input_text = f"Question:\n{row['question']}\n\nEvidence:\n{row['evidence']}"

                    out_record = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": content,
                        "metadata": {
                            "financebench_id": financebench_id,
                        }
                    }

                    f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    processed += 1

    print(f"{processed} traces saved to {FINE_TUNE_JSONL_PATH}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "split",
            "generate-batch-input",
            "submit-batch",
            "debug-batch",
            "chunk-batch-input",
            "download-batch-results",
            "convert-output",
        ],
    )
    parser.add_argument("--batch-id", type=str, help="Batch id for status checks.")
    parser.add_argument("--output-file-id", type=str, help="Output file id from the batch.")
    parser.add_argument("--batch-file", type=Path, help="Path to a batch JSONL file.")
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        help=f"Max tokens per chunk (default {MAX_BATCH_TOKENS}).",
    )
    args = parser.parse_args()

    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    else:
        print("OPENAI_API_KEY is set")
    openai = OpenAI(api_key=openai_api_key)

    if args.command == "split":
        split_and_save()
    elif args.command == "generate-batch-input":
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        generate_batch_input(train_df)
    elif args.command == "chunk-batch-input":
        batch_file = args.batch_file or BATCH_INPUT_PATH
        max_batch_tokens = args.max_batch_tokens or MAX_BATCH_TOKENS
        print(f"Splitting {batch_file} into chunks under {max_batch_tokens} tokens...")
        chunks = split_batch_file(batch_file, BATCHES_INPUT_DIR, max_batch_tokens)
        print("Chunks created:")
        for chunk in chunks:
            print(f"- {chunk}")
    elif args.command == "submit-batch":
        submit_batch(openai, args.batch_file)
    elif args.command == "download-batch-results":
        if args.output_file_id:
            output_file_id = args.output_file_id
            batch_id = args.batch_id  # may be None
        elif args.batch_id:
            batch = openai.batches.retrieve(args.batch_id)
            if not batch.output_file_id:
                raise ValueError("Batch has no output_file_id yet.")
            output_file_id = batch.output_file_id
            batch_id = batch.id
        else:
            raise ValueError("Provide either --output-file-id or --batch-id")
        download_batch_results(openai, output_file_id, batch_id=batch_id)
    elif args.command == "debug-batch":
        debug_batch(openai, args.batch_id, args.batch_file)
    elif args.command == "convert-output":
        convert_batch_output()
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()