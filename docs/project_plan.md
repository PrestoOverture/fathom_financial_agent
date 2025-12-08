# Project Roadmap: The Fathom Financial Agent

**Total Duration:** 12 Weeks
**Current Status:** Phase 1 (Week 1)

---

## Phase 1: Data Curation & The "Golden Dataset" (Weeks 1-3)

**Goal:** Create the training data. "Garbage in, garbage out." (Rule 9: Look at Traces)

### Week 1: Setup & Data Acquisition

* Set up GitHub Repo (`fathom_financial_agent`) with poetry or uv for dependency management.
* Download FinanceBench dataset from HuggingFace.
* Filter dataset for "Reasoning" type questions (discard simple retrieval).

### Week 2: Trace Generation (The Distillation)

* Write a script to format FinanceBench samples into prompts for GPT-4o.
* Use OpenAI Batch API to generate "Chain of Thought" reasoning traces.
  Prompt strategy: "You are a financial expert. Explain step-by-step..."
* Budget Check: Ensure Batch API spend is <$15.

### Week 3: Data Formatting & Verification

* Manually inspect 20 random traces to ensure quality (Rule 9).
* Convert data to Alpaca or ShareGPT JSONL format for Unsloth.
* Create a separate "Hold-out" test set for final evaluation.

---

## Phase 2: The Scientist – Fine-Tuning (Weeks 4-5)

**Goal:** Create a small model that mimics the large model's reasoning. (Rule 6 -> Reduce)

### Week 4: Training with Unsloth

* Open Google Colab (Free Tier).
* Install Unsloth.
* Load Llama-3.2-3B-Instruct.
* Run LoRA fine-tuning on the "Reasoning Traces" dataset.
* Save adapters to Hugging Face.

### Week 5: Validation

* Run the fine-tuned model against the Hold-out set.
* Compare results against the base Llama 3.2 model.
* Iterate/Re-train if necessary (adjust learning rate, epochs).

---

## Phase 3: The Engineer – Architecture & Agent (Weeks 6-9)

**Goal:** Build the RAG engine. (Rule 3: Workflow over Autonomy)

### Week 6: The RAG Pipeline

* Implement LlamaParse to ingest PDF 10-Ks.
* Build the Vector Store (pgvector) for the parsed chunks.

### Week 7: LangGraph Orchestration

* Define the Graph State (Question, Documents, Reasoning, Answer).
* Node 1: `retrieve_financial_context`.
* Node 2: `generate_reasoning` (calls your Fine-tuned Model).

### Week 8: Advanced Flow

* Add Node 3: `verify_math` (Optional Python tool to check calculations).
* Connect the graph edges.

### Week 9: Backend API

* Wrap the LangGraph agent in a FastAPI endpoint.
* Implement Streaming (SSE) to send token-by-token output.

---

## Phase 4: The Product – UI & Launch (Weeks 10-12)

**Goal:** Polish and Presentation.

### Week 10: Frontend

* Initialize Next.js project with Vercel AI SDK.
* Build a clean chat interface.
* Connect to FastAPI backend.

### Week 11: Visualization & Evaluation

* Display the "Reasoning Trace" in the UI (e.g., a collapsible "View Thinking" section).
* Run final evaluation metrics on the full test set.

### Week 12: Documentation & Demo

* Write README.md with architecture diagrams.
* Record a 2-minute Loom video demonstrating a complex financial query.
* Publish!
