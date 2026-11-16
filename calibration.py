"""Prepare blinded human labels or compare both judge prompts with them."""

import argparse
import json
import os
import random
from collections import defaultdict, deque
from pathlib import Path

from evaluator import judge_response
from main import read_jsonl


def prepare(records, count, seed):
    if not 50 <= count <= 60:
        raise ValueError("Calibration requires 50 to 60 outputs")
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = (row["model"], row["case"]["id"], row["repeat"])
        if key in seen:
            raise ValueError(f"Duplicate response: {key}")
        seen.add(key)
        if row["verdict"] == "INCONCLUSIVE":
            continue
        groups[row["case"]["category"]].append(row)
    if sum(len(group) for group in groups.values()) < count:
        raise ValueError(f"Need at least {count} distinct scored outputs, excluding inconclusive trials")
    rng = random.Random(seed)
    queues = []
    for category in sorted(groups):
        rng.shuffle(groups[category])
        queues.append(deque(groups[category]))
    selected = []
    while len(selected) < count:
        for queue in queues:
            if queue and len(selected) < count:
                row = queue.popleft()
                selected.append({
                    "model": row["model"], "case": row["case"], "repeat": row["repeat"],
                    "response": row["response"], "human_verdict": "", "reviewer": "",
                    "label_reason": "",
                })
    return selected


def agreement(human, judged):
    if len(human) != len(judged) or not human:
        raise ValueError("Human and judge labels must be non-empty and aligned")
    labels = ("PASS", "FAIL")
    if any(label not in labels for label in human + judged):
        raise ValueError("Labels must be PASS or FAIL")
    matrix = [[sum(h == a and j == b for h, j in zip(human, judged)) for b in labels] for a in labels]
    n = len(human)
    observed = (matrix[0][0] + matrix[1][1]) / n
    expected = sum(human.count(label) * judged.count(label) for label in labels) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected != 1 else None
    return {"labels": list(labels), "confusion_matrix_human_rows_judge_columns": matrix,
            "n": n, "agreement": observed, "cohen_kappa": kappa}


def validate_labels(rows, judge):
    if not 50 <= len(rows) <= 60:
        raise ValueError("Calibration requires 50 to 60 human-labeled responses")
    seen = set()
    for row in rows:
        key = (row["model"], row["case"]["id"], row["repeat"])
        if key in seen:
            raise ValueError(f"Duplicate labeled response: {key}")
        seen.add(key)
        if row["model"] == judge:
            raise ValueError("Judge model must differ from each tested model")
        if row.get("human_verdict") not in ("PASS", "FAIL"):
            raise ValueError(f"Missing human PASS/FAIL label for {key}")
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise ValueError(f"Missing reviewer for {key}")
        if not isinstance(row.get("label_reason"), str) or not row["label_reason"].strip():
            raise ValueError(f"Missing label rationale for {key}")
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            raise ValueError(f"Missing response for {key}")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample", help="Create a blinded labeling file")
    sample.add_argument("--results", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--count", type=int, default=56)
    sample.add_argument("--seed", type=int, default=42)
    compare = subparsers.add_parser("compare", help="Judge the same human-labeled responses twice")
    compare.add_argument("--labels", required=True)
    compare.add_argument("--judge", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "sample":
        rows = prepare(read_jsonl(args.results), args.count, args.seed)
        write_jsonl(args.output, rows)
        print(f"Wrote {len(rows)} blinded responses to {args.output}")
        return
    rows = read_jsonl(args.labels)
    validate_labels(rows, args.judge)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required for judge calibration")
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    judged = []
    for row in rows:
        before = judge_response(client, args.judge, row["case"], row["response"], "baseline")
        after = judge_response(client, args.judge, row["case"], row["response"], "refined")
        judged.append({**row, "baseline_judgment": before, "refined_judgment": after})
    human = [row["human_verdict"] for row in judged]
    report = {
        "judge_model": args.judge,
        "baseline": agreement(human, [row["baseline_judgment"]["verdict"] for row in judged]),
        "refined": agreement(human, [row["refined_judgment"]["verdict"] for row in judged]),
        "responses": judged,
    }
    with Path(args.output).open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    for version in ("baseline", "refined"):
        print(f"{version}: {json.dumps(report[version])}")


if __name__ == "__main__":
    main()
