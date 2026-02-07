from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILES = [
    ROOT_DIR / "results" / "test_set" / "baseline_holdout_results_original.json",
    ROOT_DIR / "results" / "test_set" / "finetuned_holdout_results_original.json",
    ROOT_DIR / "results" / "test_set" / "baseline_holdout_results_llamaparse.json",
    ROOT_DIR / "results" / "test_set" / "finetuned_holdout_results_llamaparse.json",
]
OUTPUT_FILE = ROOT_DIR / "results" / "evals" / "correctness_eval.json"


class JudgeResult(BaseModel):
    verdict: Literal["CORRECT", "INCORRECT", "REFUSED"] = Field(
        description="Binary correctness or refusal."
    )
    rationale: str = Field(description="Brief reason for the verdict.")


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def judge_answer(
    client: OpenAI, question: str, teacher_answer: str, pred_answer: str
) -> JudgeResult:
    system_prompt = (
        "You are a strict financial QA evaluator. "
        "Respond with exactly one word: CORRECT, INCORRECT, or REFUSED."
    )
    
    user_prompt = f"""
Question:
{question}

Gold Answer:
{teacher_answer}

Model Output:
{pred_answer}

Classification rules:
- CORRECT: Model's final answer matches the gold answer. Allow minor formatting differences (e.g., "$1.5B" vs "$1,500M", rounding within ±1%).
- INCORRECT: Model commits to an answer that contradicts the gold answer.
- REFUSED: Model explicitly states it cannot answer (e.g., "I don't have access", "cannot determine", "insufficient information").

"""
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=JudgeResult,
        temperature=0,
    )
    return response.choices[0].message.parsed


def evaluate_file(client: OpenAI, path: Path) -> dict[str, Any]:
    rows = load_json(path)
    details: list[dict[str, Any]] = []
    counts = {"CORRECT": 0, "INCORRECT": 0, "REFUSED": 0}

    for idx, row in enumerate(rows, start=1):
        question = row.get("question", "")
        teacher_answer = row.get("teacher_answer", "")
        pred_answer = row.get("pred_answer", "")
        item_id = row.get("id")

        print(f"[{idx}/{len(rows)}] Judging: {item_id}")
        judgment = judge_answer(client, question, teacher_answer, pred_answer)
        counts[judgment.verdict] += 1

        details.append(
            {
                "id": item_id,
                "verdict": judgment.verdict,
                "rationale": judgment.rationale,
            }
        )

    return {
        "file": str(path),
        "total": len(rows),
        "counts": counts,
        "details": details,
    }


def main() -> None:
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=openai_api_key)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "eval": "correctness",
        "files": [evaluate_file(client, path) for path in INPUT_FILES],
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Wrote correctness results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
