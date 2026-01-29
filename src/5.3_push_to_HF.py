import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

LOCAL_ADAPTER_PATH = str(Path(__file__).resolve().parents[1] / "qlora_adapters")
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
HF_MERGED_REPO_ID = os.getenv("HF_MERGED_REPO_ID")


def merge_and_push():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not found。")
        return

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            token=hf_token,
            device_map="cpu", 
            torch_dtype=torch.float16,
            return_dict=True
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
        model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER_PATH)
        model = model.merge_and_unload()

        model.push_to_hub(HF_MERGED_REPO_ID, token=hf_token, max_shard_size="2GB", safe_serialization=True)
        tokenizer.push_to_hub(HF_MERGED_REPO_ID, token=hf_token)

    except Exception as e:
        print(f"Merge Failed: {e}")

if __name__ == "__main__":
    merge_and_push()