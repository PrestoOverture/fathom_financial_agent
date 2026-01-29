import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi


def get_repo_id() -> str:
    repo_id = os.getenv("HF_REPO_ID")
    if repo_id:
        return repo_id
    raise ValueError("Missing repo id.")


def main():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    repo_id = get_repo_id()
    adapter_dir = Path(__file__).resolve().parents[1] / "qlora_adapters"
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(adapter_dir),
        path_in_repo=".",
        repo_type="model",
    )
    print(f"Uploaded adapter from {adapter_dir} to {repo_id}")


if __name__ == "__main__":
    main()