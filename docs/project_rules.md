# Tech Stack & Project Rules
## 1. Authorized Tech Stack
Strictly adhere to these tools. Do not suggest alternatives unless critical.

### Core AI & Backend
- **Orchestration:** LangGraph (Python)  
- **RAG Framework:** LlamaIndex  
- **Database (Vector + Relational):** Neon (Serverless PostgreSQL) with `pgvector` extension  
- **Parser:** LlamaParse (API)  
- **API Layer:** FastAPI (Python)  
- **Streaming:** Server-Sent Events (SSE)

### Models & Training
- **Teacher Model (Data Gen):** GPT-4o (OpenAI)  
  - Constraint: Must use Batch API  
- **Student Model (Runtime):** Llama 3.2 3B Instruct  
- **Fine-Tuning Library:** Unsloth  
  - Constraint: Run training on Google Colab T4 (Free)  
- **Technique:** LoRA (Low-Rank Adaptation) + 4-bit Quantization

### Deployment & Hosting
- **Frontend:** Vercel (Next.js)  
- **Backend Logic:** Render or Railway (FastAPI)  
- **Model Hosting (Inference):** Together AI
  - Constraint: Do not serve from Colab

## 2. Budget Constraints
- **Total Budget:** < $100 USD  
- **Data Generation:** Max $20 (OpenAI Batch API)  
- **Training Compute:** $0 (Google Colab Free Tier)  
- **Model Hosting:** ~$10–20 (Pay-as-you-go inference)  
- **Database:** $0 (Neon Free Tier: 0.5GB is sufficient for text vectors)

## 3. Coding Standards
- **Python:** Type hints required (`def func(x: int) -> str:`).  
- **Comments:** Explain *why*, not just *what* (especially for complex RAG logic).  
- **Environment:** Use `.env` for all API keys; never commit secrets.
