import csv
import os
from datetime import datetime

SCORE_FIELDS = [
    "safety_score",
    "fairness_score",
    "helpfulness_score",
    "clarity_score",
    "consistency_score",
    "overall_score",
]

CSV_FIELDS = [
    "test_id",
    "category",
    "risk_level",
    "prompt",
    "ai_response",
    "safety_score",
    "fairness_score",
    "helpfulness_score",
    "clarity_score",
    "consistency_score",
    "overall_score",
    "pass_fail",
    "flags",
    "summary",
    "timestamp",
]


def save_to_csv(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"qa_results_{timestamp}.csv")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    return filepath


def print_summary(results):
    total = len(results)
    passed = sum(1 for r in results if r["pass_fail"] == "PASS")
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0

    print("\n" + "=" * 60)
    print("           AI QA EVALUATION SUMMARY REPORT")
    print("=" * 60)
    print(f"  Total Tests Run : {total}")
    print(f"  Passed          : {passed}")
    print(f"  Failed          : {failed}")
    print(f"  Pass Rate       : {pass_rate:.1f}%")
    print("-" * 60)
    print("  AVERAGE SCORES BY DIMENSION")
    print("-" * 60)

    for field in SCORE_FIELDS:
        values = [r[field] for r in results if isinstance(r[field], (int, float))]
        avg = sum(values) / len(values) if values else 0
        bar = _score_bar(avg)
        label = field.replace("_score", "").replace("_", " ").title()
        print(f"  {label:<20} {avg:>4.1f}/10  {bar}")

    print("-" * 60)
    print("  RESULTS BY TEST CASE")
    print("-" * 60)

    for r in results:
        status = "✅ PASS" if r["pass_fail"] == "PASS" else "❌ FAIL"
        print(f"  [{r['test_id']}] {r['category']:<30} {status}  Overall: {r['overall_score']}/10")
        if r["flags"] and r["flags"].lower() != "none":
            print(f"          ⚠ Flags: {r['flags']}")

    if failed > 0:
        print("-" * 60)
        print("  FAILED TEST SUMMARIES")
        print("-" * 60)
        for r in results:
            if r["pass_fail"] == "FAIL":
                print(f"  [{r['test_id']}] {r['category']}")
                print(f"          {r['summary']}")

    print("=" * 60)


def _score_bar(score):
    filled = round(score)
    return "█" * filled + "░" * (10 - filled)
