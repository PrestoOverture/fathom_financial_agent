from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set")
else:
    print("OPENAI_API_KEY is set")
openai = OpenAI(api_key=openai_api_key)

batches = openai.batches.list(limit=20)
for b in batches.data:
    print(b.id, b.status, b.request_counts, b.model)

