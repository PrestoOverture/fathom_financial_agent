import json
import os
import re

import httpx
from dotenv import load_dotenv

from graph.state import AgentState
from langgraph.config import get_stream_writer

load_dotenv(override=True)

# non-streaming endpoint
MODAL_LLM_ENDPOINT = os.getenv("MODAL_LLM_ENDPOINT")

# token streaming endpoint (JSONL)
MODAL_STREAM_ENDPOINT = os.getenv("MODAL_STREAM_ENDPOINT")

# base model non-streaming endpoint
MODAL_LLM_BASE_ENDPOINT = os.getenv("MODAL_LLM_BASE_ENDPOINT")

# base model token streaming endpoint (JSONL)
MODAL_STREAM_BASE_ENDPOINT = os.getenv("MODAL_STREAM_BASE_ENDPOINT")

def extract_section(text: str, label: str) -> str:
    pattern = rf"{label}\s*:\s*(.*?)(?=\n\s*(?:Reasoning|Answer)\s*:|$)"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


async def reason(state: AgentState) -> dict:
    prompt = f"""
        You are a financial analyst. Answer the user's question based on the context provided.
        You MUST use this exact format:
        Reasoning:
        1. ...
        2. ...
        ...
        Answer:
        ...

        Context:
        {state.evidence}

        Question:
        {state.question}

        Your should only output the Reasoning and Answer blocks, no other text.
    """

    # get stream writer for custom events
    writer = get_stream_writer()

    accumulated_text = ""

    # async streaming from Modal
    async with httpx.AsyncClient(timeout=120.0) as client:  
        async with client.stream(
            "POST",
            MODAL_STREAM_ENDPOINT,
            json={
                "prompt": prompt,
                "max_new_tokens": 1024,
                "temperature": 0.1,
                "do_sample": False,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    delta = chunk.get("delta", "")
                    accumulated_text += delta

                    # emit custom event for SSE streaming
                    writer({"event": "reasoning_delta", "delta": delta})
                except json.JSONDecodeError:
                    continue

    # parse reasoning and answer
    reasoning = extract_section(accumulated_text, "Reasoning")
    answer = extract_section(accumulated_text, "Answer")

    return {
        "answer": answer,
        "reasoning_logs": reasoning,
        "raw_output": accumulated_text,
    }
