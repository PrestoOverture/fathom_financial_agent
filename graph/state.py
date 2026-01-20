from pydantic import BaseModel, Field
from typing import List, Optional

class AgentState(BaseModel):
    # inputs
    question: str = Field(..., description="The user's original financial query.")

    # RAG data
    retrievd_nodes: List[dict] = Field(default_factory=list, 
        description="Raw chunks retrieved from the Vector Store (text + metadata).")
    evidence: str = Field(default="", description="The formatted string evidence passed to LLM.")
    
    # outputs
    reasoning_logs: List[str] = Field(default_factory=list, description="Captured chain-of-thought steps from the <reasoning> blocks.")
    answer: str = Field(default="", description="The final extracted answer from the <answer> blocks.")

    # error handling
    raw_output: str = Field(default="", description="The raw output from the LLM, stored for debugging purposes.")