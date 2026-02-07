from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "results" / "test_set" / "retrieval_context.json"
OUTPUT_FILE = ROOT_DIR / "results" / "evals" / "retrieval_eval.json"


class ChunkJudgment(BaseModel):
    verdict: Literal["YES", "NO"] = Field(description="YES if chunk is sufficient.")
    rationale: str = Field(description="Brief reason for the verdict.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def judge_chunk(
    client: OpenAI, question: str, chunk_text: str
) -> ChunkJudgment:
    system_prompt = (
        "You are a strict retrieval evaluator. Determine if the provided chunk "
        "alone contains sufficient information to answer the question. "
        "Answer YES or NO."
    )
    user_prompt = f"""
Question:
{question}

Chunk:
{chunk_text}

Does this chunk contain information sufficient to answer the question?
"""
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ChunkJudgment,
        temperature=0,
    )
    return response.choices[0].message.parsed


def evaluate_retrievals(client: OpenAI, data: dict[str, Any]) -> dict[str, Any]:
    retrievals = data.get("retrievals", [])
    details: list[dict[str, Any]] = []
    hit_count = 0

    for idx, item in enumerate(retrievals, start=1):
        question = item.get("question", "")
        item_id = item.get("financebench_id")
        chunks = item.get("retrieved_chunks", [])

        print(f"[{idx}/{len(retrievals)}] Evaluating: {item_id}")
        chunk_results: list[dict[str, Any]] = []
        has_yes = False

        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_text = chunk.get("text", "")
            judgment = judge_chunk(client, question, chunk_text)
            if judgment.verdict == "YES":
                has_yes = True
            chunk_results.append(
                {
                    "rank": chunk_idx,
                    "verdict": judgment.verdict,
                    "rationale": judgment.rationale,
                }
            )

        if has_yes:
            hit_count += 1

        details.append(
            {
                "id": item_id,
                "has_recall_at_5": has_yes,
                "chunk_results": chunk_results,
            }
        )

    total = len(retrievals)
    recall_at_5 = (hit_count / total) if total else 0.0

    return {
        "total_questions": total,
        "hit_count": hit_count,
        "recall_at_5": recall_at_5,
        "details": details,
    }


def main() -> None:
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    client = OpenAI(api_key=openai_api_key)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(INPUT_FILE)
    results = {
        "eval": "retrieval_recall_at_5",
        "source_file": str(INPUT_FILE),
        "results": evaluate_retrievals(client, data),
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Wrote retrieval eval to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
