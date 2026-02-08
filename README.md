 # Fathom Financial Agent

> An AI agent that performs structured reasoning on complex financial tables in 10-K reports with production cost under 50¢. 

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://fathomfinancialagent.vercel.app)
[![Built with LangGraph](https://img.shields.io/badge/orchestration-LangGraph-blue)]()
[![Fine-tuned Llama 3.2](https://img.shields.io/badge/model-Llama%203.2%203B-orange)]()

![Demo GIF](assets/demo.gif)
<!-- TODO: Add screen recording of the system in action -->

---

## The Problem

Generic RAG systems fail at financial analysis because text-based PDF parsers flatten financial tables into unstructured text, destroying row/column relationships. They retrieve text but cannot calculate derived metrics (e.g., "What was the YoY change in operating margin?"). This project aim to address both problems: LlamaParse preserves table structure during ingestion, and a fine-tuned Llama 3.2 3B model performs explicit step-by-step reasoning and calculation over the retrieved evidence.

<!-- 
TODO: Consider adding a concrete example showing a failed generic RAG response vs. Fathom's correct response
-->

---

## The Solution

1. Filtered data with "reasoning" label as well as data with no label but passed LLM judge's approvel.
2. Distilled **GPT-4o reasoning traces** on the training set using batch API.
3. Built a validation pipeline to ensure distilled reasoning traces follow the correct reasoning format and no data leakage from the training set to the probe/dev/test sets.
4. Finetuned **Llama 3.2 3B** using 4-bit LoRA via Unsloth on Google Colab (Tesla T4).
5. Deployed the finetuned model on **Modal**.
6. Set up **Neon (PostgreSQL)** with pgvector and implemented **LlamaParse** for ingestion.
7. Developed the **LangGraph** workflow.
8. Integrated the data pipelines with **FastAPI Backend**.
9. Built a **Next.js** UI.
10. Deployed frontend on **Vercel** and backend on **Render**.

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────────┐
│   Browser   │────▶│   Vercel    │────▶│           Render                │
│  (Next.js)  │◀────│  (Frontend) │◀────│         (FastAPI)               │
└─────────────┘ SSE └─────────────┘     └───────────┬─────────────────────┘
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                              ┌──────────┐   ┌──────────┐   ┌──────────┐
                              │   Neon   │   │  Modal   │   │ LangGraph│
                              │ pgvector │   │  (GPU)   │   │ Workflow │
                              └──────────┘   └──────────┘   └──────────┘
                               Retrieval    Fine-tuned LLM   Orchestration
```

<!-- 
TODO: Replace ASCII diagram with a proper image if desired
-->

### LangGraph Workflow

```
[User Query] → [Retrieve] → [Reason] → [Verify] → [Response]
                   │            │          │
                   ▼            ▼          ▼
              Neon/pgvector  Llama 3.2   Math Check
              (table-aware)  (fine-tuned) (tool-based)
```

---

## Results

<!-- 
TODO: Fill in after running evaluation
-->

**Format Adherence:** Is the model following the correct reasoning format?

| Run | Valid / Total | Rate |
| --- | --- | --- |
| Baseline (original) | 0 / 15 | 0.0% |
| Finetuned (original) | 7 / 15 | 46.7% |
| Baseline (LlamaParse) | 9 / 15 | 60.0% |
| Finetuned (LlamaParse) | 13 / 15 | 86.7% |

**Correctness (GPT-4o-mini judge):** Is the final answer correct? (Results are based on shared vector store)

| Run | Correct | Incorrect | Refused | Accuracy |
| --- | --- | --- | --- | --- |
| Baseline (original) | 6 | 9 | 0 | 40.0% |
| Finetuned (original) | 4 | 11 | 0 | 26.7% |
| Baseline (LlamaParse) | 4 | 9 | 2 | 26.7% |
| Finetuned (LlamaParse) | 4 | 11 | 0 | 26.7% |

**Retrieval Recall@5 (LlamaParse only):** At least one of the top-5 chunks was judged sufficient to answer the question.

- Hit count: 5 / 15  
- Recall@5: **33.3%**
---

## Tech Stack

| Layer | Stack |
|-------|------------|
| Frontend | Next.js 14, Tailwind CSS, TypeScript |
| Backend | FastAPI, Server-Sent Events |
| Orchestration | LangGraph |
| Vector Store | Neon (PostgreSQL + pgvector) |
| Document Parsing | LlamaParse |
| Model Inference | Modal (Serverless GPU) |
| Fine-tuning | Unsloth, LoRA, Google Colab T4 |
| Base Model | Llama 3.2 3B Instruct |
| Teacher Model | GPT-4o |
| Judge | GPT-4o-mini |

---

## Project Structure

```
fathom-financial-agent/
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── schemas.py
│   └── sse.py
├── graph/                  # LangGraph workflow
│   ├── nodes/
│   │   ├── retrieve.py
│   │   ├── reason.py
│   │   └── verify.py
│   ├── state.py
│   └── workflow.py
├── frontend/               # Next.js application
├── src/                    # Data pipeline scripts
│   ├── 1_filter_data.py
│   ├── 2_audit_data.py
│   ├── 3_generate_traces.py
│   ├── 4_validate_traces.py
│   ├── 6_ingest.py
│   ├── ... the rest of the pipelines
├── data/                   # Training data & caches
└── results/                # Evaluation outputs
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- API keys: OpenAI, LlamaCloud, Modal, Neon

### Project Setup

```bash
# Clone the repository
git clone https://github.com/prestooverture/fathom-financial-agent.git
cd fathom_financial_agent

# Install dependencies
uv sync

# Set environment variables
cp .env.example .env
# Don't forget to add .env with your API keys

# Run the backend
uvicorn api.main:app --reload

# Run the frontend
cd frontend && npm run dev
```

---

## How It Works

### 1. Document Ingestion

PDFs are processed through LlamaParse, which uses vision models to reconstruct financial tables as structured Markdown. This preserves the row/column relationships that are critical for accurate retrieval.

### 2. Query Processing

When a user asks a question:

1. **Retrieve** — Semantic search over pgvector finds relevant chunks, prioritizing table nodes
2. **Reason** — The fine-tuned Llama 3.2 3B generates a structured reasoning trace with explicit calculation steps
3. **Verify** — A Python-based math engine checks any arithmetic in the response against the stated inputs

### 3. Streaming Response

Results stream back to the user in real-time via SSE, with the reasoning trace shown in a collapsible panel.

---

## Lessons Learned

<!-- 
TODO: Expand these into the blog post
-->

1. **The Parameter Wall** — A 3B model can learn *how* to solve financial problems but lacks the internal capacity to reliably execute arithmetic. Solution: tool-augmented reasoning.

2. **Distribution Shift Matters** — The RAG prompt structure must match the fine-tuning data structure exactly, or the model's "muscle memory" breaks.

3. **Table Parsing is the Bottleneck** — Without LlamaParse, retrieval quality collapsed. Standard PDF parsers destroy the structure that makes financial tables meaningful.

---

## Limitations & Future Work

**Current Limitations:**
- Single-document queries only (no cross-company comparisons)
- Limited to 10-K annual reports
- Verification loop catches arithmetic errors but not logical errors

**Future Improvements:**
- Multi-document retrieval for comparative analysis
- Support for 10-Q quarterly reports and 8-K filings
- Fine-tuned embedding model for financial domain
- Confidence scoring for answers

---

## Acknowledgments

[FinanceBench](https://github.com/patronus-ai/financebench) by Patronus AI — The benchmark dataset that made rigorous evaluation possible
