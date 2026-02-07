from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from graph.workflow import create_graph

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT_DIR / "data" / "test" / "test.jsonl"
OUTPUT_FILE = ROOT_DIR / "results" / "test_set" / "finetuned_holdout_results_llamaparse1.json"


def load_test_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


async def run_eval() -> None:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    graph = create_graph()
    questions = load_test_questions(TEST_FILE)

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(questions, start=1):
        question = item.get("question", "")
        financebench_id = item.get("financebench_id")
        teacher_answer = item.get("answer")

        print(f"[{idx}/{len(questions)}] Running: {financebench_id}")
        output = await graph.ainvoke({"question": question})

        results.append(
            {
                "id": financebench_id,
                "question": question,
                "teacher_answer": teacher_answer,
                "pred_answer": output.get("raw_output"),
            }
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Saved eval results to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(run_eval())
