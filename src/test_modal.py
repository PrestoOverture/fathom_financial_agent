import os
import requests

URL = "https://prestooverture--fathom-llama32-3b-llm-generate.modal.run"

payload = {
    "prompt": "Respond exactly with <answer>Hello</answer>",
    "max_new_tokens": 8,
    "temperature": 0.0,
    "do_sample": False,
}

r = requests.post(URL, json=payload, timeout=900)
print("status:", r.status_code)
print(r.text)
