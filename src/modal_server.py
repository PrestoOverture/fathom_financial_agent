import os
import modal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = True

APP_NAME = "fathom-llama32-3b"
HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_ID = "PrestoOverture/fathom-llama-3b-merged"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "safetensors",
        "fastapi",
    )
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.cls(
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=600,
    scaledown_window=5,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class LLM:
    @modal.enter(snap=True)
    def load_model(self):
        """
        Don't change this function. It's used for GPU snapshot.
        Any changes will break the snapshot and force a 90s cold start.
        """
        print("GPU SNAPSHOT: load_model running")
        os.environ["HF_HOME"] = HF_CACHE_PATH

        print("load_model: loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ["HF_TOKEN"])

        print("load_model: loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=os.environ["HF_TOKEN"],
            dtype=torch.float16,
        ).to("cuda")

        self.model.eval()

        warmup = self.tokenizer("warmup", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = self.model.generate(**warmup, max_new_tokens=1)
        
        hf_cache.commit()

    @modal.fastapi_endpoint(method="POST")
    def generate(self, req: GenerateRequest) -> dict:
        prompt = req.prompt
        max_new_tokens = req.max_new_tokens
        temperature = req.temperature
        do_sample = req.do_sample

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # if text.startswith(prompt):
        #     text = text[len(prompt):]
        text = text.lstrip()
        prompt_stripped = prompt.lstrip()
        if text.startswith(prompt_stripped):
            text = text[len(prompt_stripped):].lstrip()

        return {"text": text}
