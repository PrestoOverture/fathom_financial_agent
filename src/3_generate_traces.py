import json
from pathlib import Path
from typing import Tuple
import pandas as pd 
from pandas import DataFrame

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE_PATH = ROOT_DIR / "data" / "audited" / "financialbench_filtered_reasoning.csv"

TRAIN_CSV_PATH = ROOT_DIR / "data" / "processed" / "financebench_train.csv"
TEST_CSV_PATH = ROOT_DIR / "data" / "test" / "financebench_test.csv"
TEST_JSONL_PATH = ROOT_DIR / "data" / "test.jsonl"

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE_PATH)
    return df

def split_train_test(df: DataFrame, n_test: int) -> Tuple[DataFrame, DataFrame]:
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = df_shuffled.iloc[:n_test].copy()
    train_df = df_shuffled.iloc[n_test:].copy()
    return train_df, test_df

def save_jsonl(test_df: DataFrame) -> None:
    test_df.to_json(TEST_JSONL_PATH, orient="records", lines=True)
    cols_to_keep = ["financebench_id", "question", "context", "answer"]
    for col in cols_to_keep:
        if col not in test_df.columns:
            raise ValueError(f"Column {col} not found in test dataframe")
    
    TEST_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TEST_JSONL_PATH.open("w", encoding="utf-8") as f:
        for _, row in test_df.iterrows():
            record = {
                "financebench_id": str(row["financebench_id"]),
                "question": str(row["question"]),
                "context": str(row["context"]),
                "answer": str(row["answer"]),
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