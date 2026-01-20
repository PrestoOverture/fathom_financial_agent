import os
import re
from dotenv import load_dotenv
from llama_index.llms.together import TogetherLLM as Together
from graph.state import AgentState

load_dotenv(override=True)

def extract_xml_tag(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def reason(state: AgentState):
    print(f"Reasoning about question: {state.question}")
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        raise ValueError("WARNING: TOGETHER_API_KEY is NOT set!")
    else:
        print("TOGETHER_API_KEY is set :)")
    
    llm = Together(model="meta-llama/Llama-3.2-3B-Instruct-Turbo", api_key=together_api_key, temperature=0.1, max_tokens=1024)
    prompt = f"""
        You are a financial analyst. Answer the user's question based on the context provided. Respond with exactly: <reasoning>...</reasoning> and <answer>...</answer>.
        
        \n\nQuestion:
        {state.question}

        \n\Evidence:\n
        {state.evidence}
    """
    
    response = llm.complete(prompt)
    raw_output = response.text

    reasoning = extract_xml_tag(raw_output, "reasoning")
    answer = extract_xml_tag(raw_output, "answer")

    return {
        "answer": answer,
        "reasoning_logs": [reasoning],
        "raw_output": raw_output,
    }