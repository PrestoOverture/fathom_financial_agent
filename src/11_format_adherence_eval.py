from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILES = [
    ROOT_DIR / "results" / "test_set" / "baseline_holdout_results_original.json",
    ROOT_DIR / "results" / "test_set" / "finetuned_holdout_results_original.json",
    ROOT_DIR / "results" / "test_set" / "baseline_holdout_results_llamaparse.json",
    ROOT_DIR / "results" / "test_set" / "finetuned_holdout_results_llamaparse.json",
]
OUTPUT_FILE = ROOT_DIR / "results" / "evals" / "format_adherence.json"

REASON_PATTERN = re.compile(r"<reasoning>\s*.*?\s*</reasoning>", flags=re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*.*?\s*</answer>", flags=re.DOTALL)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_file(path: Path) -> dict[str, Any]:
    rows = load_json(path)
    details: list[dict[str, Any]] = []
    valid_count = 0

    for row in rows:
        pred_answer = row.get("pred_answer") or ""
        reason_matches = REASON_PATTERN.findall(pred_answer)
        answer_matches = ANSWER_PATTERN.findall(pred_answer)

        reason_count = len(reason_matches)
        answer_count = len(answer_matches)
        is_valid = reason_count == 1 and answer_count == 1
        if is_valid:
            valid_count += 1

        details.append(
            {
                "id": row.get("id"),
                "reason_count": reason_count,
                "answer_count": answer_count,
                "is_valid": is_valid,
            }
        )

    total = len(rows)
    return {
        "file": str(path),
        "total": total,
        "valid": valid_count,
        "invalid": total - valid_count,
        "details": details,
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "eval": "format_adherence",
        "files": [evaluate_file(path) for path in INPUT_FILES],
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Wrote format adherence results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
