import os
import re
import requests
from dotenv import load_dotenv
from graph.state import AgentState

load_dotenv(override=True)

# MODAL_LLM_ENDPOINT = os.environ.get(
#     "MODAL_LLM_ENDPOINT",
#     "https://prestooverture--fathom-llama32-3b-llm-generate.modal.run/generate",
# )

MODAL_LLM_ENDPOINT = "https://prestooverture--fathom-llama32-3b-llm-generate.modal.run"

def extract_xml_tag(text: str, tag: str) -> str:
    # pattern = f"<{tag}>(.*?)</{tag}>"
    # match = re.search(pattern, text, re.DOTALL)
    # return match.group(1).strip() if match else ""
    pattern = rf"<{tag}\s*>(.*?)</{tag}\s*>"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else ""

def call_modal_llm(prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "do_sample": False,
    }

    response = requests.post(MODAL_LLM_ENDPOINT, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()["text"]

def reason(state: AgentState):
    print(f"Reasoning about question: {state.question}")
    prompt = f"""
        You are a financial analyst. Answer the user's question based on the context provided. 
        You MUST use this exact format:
        <reasoning>
        1. ...
        2. ...
        ...
        </reasoning>
        <answer>
        Final answer here.
        </answer>

        ### Input:
        Context: 
        {state.evidence}

        Question: 
        {state.question}
    """

    text = call_modal_llm(prompt)

    reasoning = extract_xml_tag(text, "reasoning")
    answer = extract_xml_tag(text, "answer")

    return {
        "answer": answer,
        "reasoning_logs": reasoning,
        "raw_output": text,
    }
