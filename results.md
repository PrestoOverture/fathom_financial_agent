# Results Summary

**Scope:** 4 holdout result files (15 questions each), plus retrieval recall@5 over the same 15 questions.

**Format Adherence**  
Check: exactly one `<reason>...</reason>` block and one `<answer>...</answer>` block.

| Run | Valid / Total | Rate |
| --- | --- | --- |
| Baseline (original) | 0 / 15 | 0.0% |
| Finetuned (original) | 7 / 15 | 46.7% |
| Baseline (LlamaParse) | 9 / 15 | 60.0% |
| Finetuned (LlamaParse) | 13 / 15 | 86.7% |

**Correctness (GPT-4o-mini judge)**  
Verdicts across 15 questions each.

| Run | Correct | Incorrect | Refused | Accuracy |
| --- | --- | --- | --- | --- |
| Baseline (original) | 6 | 9 | 0 | 40.0% |
| Finetuned (original) | 4 | 11 | 0 | 26.7% |
| Baseline (LlamaParse) | 4 | 9 | 2 | 26.7% |
| Finetuned (LlamaParse) | 4 | 11 | 0 | 26.7% |

**Retrieval Recall@5 (LlamaParse only)**  
At least one of the top-5 chunks was judged sufficient to answer the question.

- Hit count: 5 / 15  
- Recall@5: **33.3%**

**Notes**
- Format adherence improves substantially on the LlamaParse finetuned run (13/15) relative to the original baseline (0/15).
- Correctness remains low across all runs, with the original baseline currently highest at 6/15.
- Retrieval recall@5 is 33.3%, suggesting retrieval is a major bottleneck before correctness can improve.
