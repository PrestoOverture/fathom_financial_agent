import json
import os
import threading

import modal
import torch
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = False


APP_NAME = "fathom-llama32-3b-base"
HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 4096

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "safetensors",
        "fastapi",
        "unsloth",
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

        print("load_model: loading model (unsloth, 4-bit)")
        import unsloth
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        self.model.eval()

        warmup = self.tokenizer("warmup", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = self.model.generate(**warmup, max_new_tokens=1)

        hf_cache.commit()

    @modal.fastapi_endpoint(method="POST")
    def generate(self, req: GenerateRequest) -> dict:
        prompt = req.prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            sequences = self.model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=req.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = sequences[0, input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).lstrip()

        return {"text": text}

    @modal.fastapi_endpoint(method="POST")
    def generate_stream(self, req: GenerateRequest):
        from fastapi.responses import StreamingResponse
        from transformers import TextIteratorStreamer

        prompt = req.prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature if req.temperature > 0 else 1.0,
            do_sample=req.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(target=self._generate_in_thread, args=(generation_kwargs,))
        thread.start()

        def token_generator():
            for token_text in streamer:
                if token_text:
                    yield json.dumps({"delta": token_text}) + "\n"
            thread.join()

        return StreamingResponse(
            token_generator(),
            media_type="application/x-ndjson",
        )

    def _generate_in_thread(self, generation_kwargs: dict):
        with torch.no_grad():
            self.model.generate(**generation_kwargs)
