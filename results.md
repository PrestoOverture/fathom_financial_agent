# Results Summary

**Scope:** 5 holdout result files (15 questions each), plus retrieval recall@5 evaluations over the same 15 questions.

---

## Format Adherence

Check: exactly one `<reason>...</reason>` block and one `<answer>...</answer>` block.


| Run                    | Valid / Total | Rate  |
| ---------------------- | ------------- | ----- |
| Baseline (original)    | 0 / 15        | 0.0%  |
| Finetuned (original)   | 7 / 15        | 46.7% |
| Baseline (LlamaParse)  | 9 / 15        | 60.0% |
| Finetuned (LlamaParse) | 13 / 15       | 86.7% |


---

## Correctness (GPT-4o-mini judge)

Verdicts across 15 questions each.


| Run                             | Correct | Incorrect | Refused | Accuracy  |
| ------------------------------- | ------- | --------- | ------- | --------- |
| Baseline (original)             | 6       | 9         | 0       | 40.0%     |
| Finetuned (original)            | 4       | 11        | 0       | 26.7%     |
| Baseline (LlamaParse)           | 4       | 9         | 2       | 26.7%     |
| Finetuned (LlamaParse)          | 4       | 11        | 0       | 26.7%     |
| **Finetuned (fixed retrieval)** | **4**   | **11**    | **0**   | **26.7%** |


---

## Retrieval Recall@5

### Baseline (pure vector similarity, no filtering)

At least one of the top-5 chunks was judged sufficient to answer the question.

- Hit count: 5 / 15
- Recall@5: **33.3%**

### Retrieval Ablation (Phase 1 — metadata filtering + reranking)


| Stage | Config                                 | Recall@5  | Hits     | Delta from baseline |
| ----- | -------------------------------------- | --------- | -------- | ------------------- |
| Base  | Pure vector, no filtering, top_k=5     | 33.3%     | 5/15     | --                  |
| A     | Metadata filtering, top_k=5            | 26.7%     | 4/15     | -6.7%               |
| B     | Metadata filtering, top_k=15           | 26.7%     | 4/15     | -6.7%               |
| C     | Metadata filtering, top_k=15, reranker | **40.0%** | **6/15** | **+6.7%**           |


### Per-Question Retrieval Detail (Stage C vs Baseline)


| Question ID           | Company         | Topic                     | Baseline | Stage C  | Change |
| --------------------- | --------------- | ------------------------- | -------- | -------- | ------ |
| financebench_id_03473 | Coca-Cola       | FY2017 ROA                | Miss     | **Hit**  | Gained |
| financebench_id_06655 | Amazon          | FY2017 DPO                | Miss     | Miss     | --     |
| financebench_id_00566 | Verizon         | Debt change 2021-2022     | Hit      | Hit      | --     |
| financebench_id_00724 | Pfizer          | Q2 2023 revenue by region | Hit      | Hit      | --     |
| financebench_id_03031 | Lockheed Martin | FY2021 NWC                | Miss     | Miss     | --     |
| financebench_id_03849 | MGM Resorts     | FY2018-2020 capex/revenue | Miss     | Miss     | --     |
| financebench_id_06272 | Coca-Cola       | FY2022 dividend payout    | Miss     | **Hit**  | Gained |
| financebench_id_00799 | AMCOR           | Quick ratio FY2022-2023   | Miss     | Miss     | --     |
| financebench_id_00605 | Ulta Beauty     | Q4 stock repurchases      | Miss     | Miss     | --     |
| financebench_id_01911 | MGM Resorts     | FY2022 interest coverage  | Miss     | Miss     | --     |
| financebench_id_00807 | 3M              | Q2 FY2023 quick ratio     | Miss     | Miss     | --     |
| financebench_id_00956 | J&J             | FY2022 high growth?       | Miss     | Miss     | --     |
| financebench_id_00917 | AMD             | FY2022 operating margin   | Hit      | **Miss** | Lost   |
| financebench_id_00464 | Boeing          | Cyclicality               | Hit      | Hit      | --     |
| financebench_id_00684 | AMCOR           | Gross margin FY2023       | Miss     | **Hit**  | Gained |


**Net: +3 gained, -1 lost = +2 net hits** (5 -> 6 correct out of 15)

---

## Analysis

**Format adherence** improves substantially after fine-tuning (0% -> 86.7% on LlamaParse), confirming the distillation successfully taught the model the structured reasoning format.

**Correctness** remains stuck at 26.7% across the LlamaParse and fixed-retrieval runs. The same 4 questions are consistently correct across runs: Verizon debt (yes/no), J&J growth (qualitative), Boeing cyclicality (qualitative), and one calculation question. The model experiences alignment tax after fine-tuning (original baseline was 40%).

**Retrieval** is the primary bottleneck:

- Metadata filtering correctly constrains to the right document for 13/15 questions, eliminating cross-document confusion.
- However, within-document chunk relevance is the real problem — financial tables (balance sheets, income statements, cash flow) don't rank high enough by embedding similarity.
- The cross-encoder reranker (BAAI/bge-reranker-base) provides the only measurable lift, promoting table chunks that embedding similarity misses.
- 9/15 questions still fail retrieval — these are chunking/embedding quality problems, not filtering problems.

**Bottleneck shift:** After Phase 1, the remaining failures are split between:

1. **Within-document chunk relevance** (9/15 retrieval misses): needs chunking strategy changes or embedding model upgrade
2. **3B model arithmetic hallucination** (even with correct chunks, the model miscalculates): needs stronger model or tool-augmented math

