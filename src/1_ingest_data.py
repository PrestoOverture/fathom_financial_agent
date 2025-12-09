import os
import pandas as pd
from datasets import load_dataset

def load_financial_data(dataset_id, output_path):
    print(f"Loading dataset from {dataset_id} to {output_path}")
    ds = load_dataset(dataset_id, split="train")
    print(f"Dataset loaded with {len(ds)} rows")
    df = pd.DataFrame(ds)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    return df

def filter_financial_data(df, output_path):
    reasoning_df = df["question_reasoning"].str.contains("reasoning", case=False, na=False)
    novel_generated_df = df['question_type'] == 'novel-generated'
    filter_df = reasoning_df | novel_generated_df
    filtered_df = df[filter_df].copy()
    print(f"Filtered dataset with {len(filtered_df)} rows")
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered dataset saved to {output_path}")

if __name__ == "__main__":
    dataset_id = "PatronusAI/financebench"
    output_dir_raw = "data/raw"
    output_path_raw = os.path.join(output_dir_raw, "financebench_raw.csv")

    if not os.path.exists(output_dir_raw):
        os.makedirs(output_dir_raw)

    if os.path.exists(output_path_raw):
        df = pd.read_csv(output_path_raw)
        print(f"Dataset loaded from {output_path_raw}")
    else:
        df = load_financial_data(dataset_id, output_path_raw)

    output_dir_filtered = "data/filtered"
    output_path_filtered = os.path.join(output_dir_filtered, "financebench_filtered.csv")

    if not os.path.exists(output_dir_filtered):
        os.makedirs(output_dir_filtered)
        
    filter_financial_data(df, output_path_filtered)