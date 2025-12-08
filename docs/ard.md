# Architecture Decision Records (ADR)
This document records the critical architectural decisions made for **The Fathom Financial Agent** project.

---

## ADR-001: PDF Parsing Strategy
**Decision:** Use LlamaParse (API).  
**Reasoning:** Standard parsers (e.g., PyPDF) destroy table structures. LlamaParse uses vision models to reconstruct financial tables into Markdown, which is essential for reasoning.

---

## ADR-002: Model Lifecycle (Train vs. Serve)
**Status:** Accepted  

### Context
We need to fine-tune a model cheaply but serve it reliably for the frontend.

### Decision
- **Train:** Use Google Colab (Free T4) with Unsloth.  
- **Serve:** Deploy the fine-tuned adapter to Together AI (or compatible inference provider).

### Reasoning
- Colab is ephemeral and not suitable for hosting APIs due to latency and disconnects.  
- Together AI and similar cloud inference providers offer optimized inference engines (vLLM) that provide higher speed and reliability than self-hosted containers.

---

## ADR-003: Model Selection
**Decision:** Llama 3.2 3B Instruct.  
**Reasoning:** The 3B model size is low-cost for fine-tuning and sufficiently capable for the project's financial reasoning requirements when distilled from GPT-4o.

---

## ADR-004: Orchestration Framework
**Decision:** LangGraph.  
**Reasoning:** The project requires a deterministic Retrieve → Reason → Verify loop. LangGraph provides state management and control-flow guarantees not available in standard chain-based frameworks.

---

## ADR-005: Frontend Streaming Protocol
**Decision:** Server-Sent Events (SSE) via FastAPI.  
**Reasoning:** SSE has native support in the Vercel AI SDK and is simpler than WebSockets for one-way token streaming.

---

## ADR-006: Vector Store & Database
**Status:** Accepted

### Context
We need to store document chunks (vectors) and potentially user logs or metadata.

### Decision
Use **Neon (PostgreSQL)** with **pgvector**.

### Alternatives Considered
- ChromaDB (Local)  
- Pinecone (Managed)

### Reasoning
- **Production Readiness:** PostgreSQL is an industry standard. pgvector allows a unified stack, avoiding the need for separate relational and vector databases.  
- **Hybrid Search:** Enables efficient metadata filtering (e.g., `WHERE year = 2023`) before running vector similarity search.  
- **Cost Efficiency:** Neon’s free tier is generous and serverless (scales to zero), reducing operational overhead.

