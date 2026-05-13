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
    # Model A
    "model_a_name",
    "model_a_response",
    "model_a_safety",
    "model_a_fairness",
    "model_a_helpfulness",
    "model_a_clarity",
    "model_a_consistency",
    "model_a_overall",
    "model_a_pass_fail",
    "model_a_flags",
    "model_a_summary",
    # Model B
    "model_b_name",
    "model_b_response",
    "model_b_safety",
    "model_b_fairness",
    "model_b_helpfulness",
    "model_b_clarity",
    "model_b_consistency",
    "model_b_overall",
    "model_b_pass_fail",
    "model_b_flags",
    "model_b_summary",
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


def print_summary(results, model_a_name, model_b_name):
    total = len(results)

    a_passed = sum(1 for r in results if r["model_a_pass_fail"] == "PASS")
    b_passed = sum(1 for r in results if r["model_b_pass_fail"] == "PASS")

    print("\n" + "=" * 70)
    print("              AI QA COMPARATIVE EVALUATION REPORT")
    print("=" * 70)
    print(f"  Total Test Cases : {total}")
    print(f"  {'Model':<35} {'Pass':>5}  {'Fail':>5}  {'Pass Rate':>10}")
    print(f"  {'-'*55}")
    print(f"  {model_a_name:<35} {a_passed:>5}  {total - a_passed:>5}  {a_passed/total*100:>9.1f}%")
    print(f"  {model_b_name:<35} {b_passed:>5}  {total - b_passed:>5}  {b_passed/total*100:>9.1f}%")

    print("\n" + "-" * 70)
    print("  AVERAGE SCORES BY DIMENSION")
    print(f"  {'Dimension':<22} {model_a_name[:18]:>18}   {model_b_name[:18]:>18}")
    print(f"  {'-'*62}")

    dim_map = [
        ("Safety",      "model_a_safety",      "model_b_safety"),
        ("Fairness",    "model_a_fairness",     "model_b_fairness"),
        ("Helpfulness", "model_a_helpfulness",  "model_b_helpfulness"),
        ("Clarity",     "model_a_clarity",      "model_b_clarity"),
        ("Consistency", "model_a_consistency",  "model_b_consistency"),
        ("Overall",     "model_a_overall",      "model_b_overall"),
    ]

    for label, a_key, b_key in dim_map:
        a_vals = [r[a_key] for r in results if isinstance(r[a_key], (int, float))]
        b_vals = [r[b_key] for r in results if isinstance(r[b_key], (int, float))]
        a_avg = sum(a_vals) / len(a_vals) if a_vals else 0
        b_avg = sum(b_vals) / len(b_vals) if b_vals else 0
        print(f"  {label:<22} {a_avg:>6.1f}/10  {_bar(a_avg):<10}   {b_avg:>6.1f}/10  {_bar(b_avg)}")

    print("\n" + "-" * 70)
    print("  RESULTS BY TEST CASE")
    print(f"  {'ID':<7} {'Category':<35} {'Model A':>10}  {'Model B':>10}")
    print(f"  {'-'*64}")

    for r in results:
        a_status = "✅ PASS" if r["model_a_pass_fail"] == "PASS" else "❌ FAIL"
        b_status = "✅ PASS" if r["model_b_pass_fail"] == "PASS" else "❌ FAIL"
        print(f"  [{r['test_id']}] {r['category']:<35} {a_status}   {b_status}")

        a_flags = r.get("model_a_flags", "none")
        b_flags = r.get("model_b_flags", "none")
        if a_flags and a_flags.lower() != "none":
            print(f"          ⚠ Model A flags: {a_flags}")
        if b_flags and b_flags.lower() != "none":
            print(f"          ⚠ Model B flags: {b_flags}")

    failures = [r for r in results if r["model_a_pass_fail"] == "FAIL" or r["model_b_pass_fail"] == "FAIL"]
    if failures:
        print("\n" + "-" * 70)
        print("  FAILURE DETAILS")
        print("-" * 70)
        for r in failures:
            if r["model_a_pass_fail"] == "FAIL":
                print(f"  ❌ [{r['test_id']}] {model_a_name} — {r['model_a_summary']}")
            if r["model_b_pass_fail"] == "FAIL":
                print(f"  ❌ [{r['test_id']}] {model_b_name} — {r['model_b_summary']}")

    print("=" * 70)


def _bar(score):
    filled = round(score)
    return "█" * filled + "░" * (10 - filled)
