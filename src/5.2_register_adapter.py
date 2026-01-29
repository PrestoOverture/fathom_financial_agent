import os
from dotenv import load_dotenv
from together import Together

load_dotenv(override=True)


def register_adapter():
    HF_REPO_ID = os.getenv("HF_REPO_ID")
    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID is not set")

    TOGETHER_MODEL_NAME = "your-adapter-name"
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("TOGETHER_API_KEY not found.")
        return

    client = Together(api_key=api_key)
    print(f"Registering '{HF_REPO_ID}' on Together AI as '{TOGETHER_MODEL_NAME}'...")
    response = client.models.upload(
        model_name=TOGETHER_MODEL_NAME,
        model_source=HF_REPO_ID,
        model_type="adapter",
        base_model="meta-llama/Llama-3.2-3B-Instruct-Turbo"
    )
    print("\nAdapter registered.")
    print(f"Model Name: {response.model_name}")
    print(f"Job ID: {response.job_id}")
        

if __name__ == "__main__":
    register_adapter()