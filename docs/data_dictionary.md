# Data Dictionary & Schemas

## 1. Raw Input Data

**Source:** SEC EDGAR Database / FinanceBench
**Format:** PDF (10-K Annual Reports)

**Key Sections:**

* Consolidated Balance Sheets
* Consolidated Statements of Operations (Income Statement)
* Consolidated Statements of Cash Flows

---

## 2. RAG Knowledge Base (LlamaIndex)

**Processing:** Parsed via LlamaParse → Markdown → Vector Embeddings.

**Schema:**

```json
{
  "node_id": "uuid",
  "text": "Markdown content of the chunk...",
  "metadata": {
    "page_number": 45,
    "document_title": "NVIDIA 2023 10-K",
    "section_type": "Table",
    "ticker": "NVDA"
  },
  "embedding": [0.012, -0.34, ...]
}
```

---

## 3. Golden Dataset (Training Data)

**Source:** FinanceBench (Subset: Reasoning Questions)
**Purpose:** Teach the model how to think, not just what to know.

**Intermediate Format (Reasoning Trace Generation):**

```json
{
  "question": "What is the FY2023 capital expenditure?",
  "context": "Table extract showing CapEx...",
  "answer": "$1.5 Billion"
}
```

---

## 4. Fine-Tuning Dataset Format (Unsloth/Alpaca)

**Format:** JSONL

**Structure:**

```json
{
  "instruction": "You are a financial analyst. Answer the user's question based on the context provided. Show your reasoning steps.",
  "input": "Context: [Markdown Table of Cash Flows]... Question: Did the company generate positive free cash flow?",
  "output": "<reasoning>
1. Identify Net Cash from Operations: $500M
2. Identify Capital Expenditures: $200M
3. Formula: FCF = OCF - CapEx
4. Calculation: 500 - 200 = 300
</reasoning>
<answer>Yes, the company generated $300M in free cash flow.</answer>"
}
```

---

## 5. Agent Output Schema (Frontend Consumption)

**Format:** Server-Sent Events (Streamed)

**Structure:**

* **Event:** `reasoning` → "Checking balance sheet..." (shown in collapsible UI)
* **Event:** `tool_call` → "Retrieving page 45..."
* **Event:** `final_response` → "The Free Cash Flow for 2023 was..."
