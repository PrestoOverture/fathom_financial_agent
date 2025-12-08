# Project Charter: The Fathom Financial Agent (Agentic RAG)

## 1. Project Overview

* **Title:** The Fathom Financial Agent.
* **Type:** Agentic RAG System for Financial Reasoning.
* **One-Liner:** An AI agent that doesn't just "chat" with PDFs but performs structured reasoning on complex financial tables in 10-K reports.

## 2. The Problem (Rule 1)

Generic RAG systems fail at financial analysis because:

* **Table Blindness:** They cannot accurately parse dense financial tables (Balance Sheets, Cash Flow).
* **Lack of Reasoning:** They retrieve text but fail to calculate derived metrics (e.g., "How did the Operating Margin change YoY?").
* **Hallucination:** They invent numbers when they cannot find them in the dense text.

## 3. The Solution

A vertical AI agent specialized in financial reasoning, built by distilling the reasoning capabilities of a Frontier Model (GPT-4o) into a small, efficient Local Model (Llama 3.2 3B).

### Core Architecture

* **Ingestion:** LlamaParse for SOTA table extraction.
* **Orchestration:** LangGraph for controlled, step-by-step workflow (Retrieve -> Reason -> Verify).
* **Brain:** A custom fine-tuned Llama 3.2 3B model trained on "Reasoning Traces" from the FinanceBench dataset.

## 4. Success Metrics (Rule 2 & 10)

* **Primary Metric (Accuracy):** The agent must accurately answer "Reasoning" questions from the FinanceBench hold-out set with >80% accuracy (evaluated by GPT-4o-mini as judge).
* **Secondary Metric (Cost):** The entire training and development process must remain under $100.
* **Latency:** End-to-end query response time under 10 seconds (excluding initial PDF parsing).

## 5. Tech Stack Constraints

* **Orchestration:** LangGraph (Python)
* **RAG Framework:** LlamaIndex
* **Parsing:** LlamaParse (Free tier)
* **LLM (Inference):** Llama 3.2 3B (Fine-tuned)
* **LLM (Data Gen):** GPT-4o (via OpenAI Batch API for 50% cost savings)
* **Fine-Tuning:** Unsloth (running on Google Colab Tesla T4 Free Tier)
* **Frontend:** Next.js + Vercel AI SDK
* **Deployment:** Vercel (Frontend) + Modal/Render/FastAPI (Backend)
* **Database:** Neon + pgvector
* **Model Hosting:** Together AI

## 6. Target Audience

* **Users:** Financial Analysts, Investors
* **Evaluators:** Recruiters looking for skills in Fine-tuning, Agentic Workflows, and Evaluation Systems
