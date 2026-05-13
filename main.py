import os
import time
from datetime import datetime

from groq import Groq

from test_cases import TEST_CASES
from evaluator import evaluate
from reporter import save_to_csv, print_summary

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Model A — smaller, faster, less safety-tuned (expected to struggle on adversarial cases)
MODEL_A = "llama-3.1-8b-instant"
# Model B — larger, more capable, better safety alignment
MODEL_B = "llama-3.3-70b-versatile"

results = []

print("=" * 70)
print("         AI QA COMPARATIVE EVALUATION FRAMEWORK")
print(f"  Model A (challenger) : {MODEL_A}")
print(f"  Model B (baseline)   : {MODEL_B}")
print(f"  Test cases           : {len(TEST_CASES)}")
print("=" * 70)

for test in TEST_CASES:
    print(f"\n[{test['id']}] {test['category']}  (risk: {test['risk_level']})")
    print(f"  Prompt: {test['prompt'][:90]}...")

    # --- Run Model A ---
    response_a = groq_client.chat.completions.create(
        model=MODEL_A,
        messages=[{"role": "user", "content": test["prompt"]}],
        temperature=0.7,
    ).choices[0].message.content

    eval_a = evaluate(groq_client, test, response_a)
    time.sleep(0.3)

    # --- Run Model B ---
    response_b = groq_client.chat.completions.create(
        model=MODEL_B,
        messages=[{"role": "user", "content": test["prompt"]}],
        temperature=0.7,
    ).choices[0].message.content

    eval_b = evaluate(groq_client, test, response_b)
    time.sleep(0.3)

    result = {
        "test_id": test["id"],
        "category": test["category"],
        "risk_level": test["risk_level"],
        "prompt": test["prompt"],
        # Model A
        "model_a_name": MODEL_A,
        "model_a_response": response_a,
        "model_a_safety":      eval_a.get("safety_score", 0),
        "model_a_fairness":    eval_a.get("fairness_score", 0),
        "model_a_helpfulness": eval_a.get("helpfulness_score", 0),
        "model_a_clarity":     eval_a.get("clarity_score", 0),
        "model_a_consistency": eval_a.get("consistency_score", 0),
        "model_a_overall":     eval_a.get("overall_score", 0),
        "model_a_pass_fail":   eval_a.get("pass_fail", "FAIL"),
        "model_a_flags":       eval_a.get("flags", "none"),
        "model_a_summary":     eval_a.get("summary", ""),
        # Model B
        "model_b_name": MODEL_B,
        "model_b_response": response_b,
        "model_b_safety":      eval_b.get("safety_score", 0),
        "model_b_fairness":    eval_b.get("fairness_score", 0),
        "model_b_helpfulness": eval_b.get("helpfulness_score", 0),
        "model_b_clarity":     eval_b.get("clarity_score", 0),
        "model_b_consistency": eval_b.get("consistency_score", 0),
        "model_b_overall":     eval_b.get("overall_score", 0),
        "model_b_pass_fail":   eval_b.get("pass_fail", "FAIL"),
        "model_b_flags":       eval_b.get("flags", "none"),
        "model_b_summary":     eval_b.get("summary", ""),
        "timestamp": datetime.now().isoformat(),
    }

    results.append(result)

    a_status = "✅ PASS" if result["model_a_pass_fail"] == "PASS" else "❌ FAIL"
    b_status = "✅ PASS" if result["model_b_pass_fail"] == "PASS" else "❌ FAIL"
    print(f"  Model A ({MODEL_A}): {a_status}  Overall: {result['model_a_overall']}/10  Safety: {result['model_a_safety']}")
    print(f"  Model B ({MODEL_B}): {b_status}  Overall: {result['model_b_overall']}/10  Safety: {result['model_b_safety']}")

csv_path = save_to_csv(results)
print(f"\n📄 Results saved to: {csv_path}")

print_summary(results, MODEL_A, MODEL_B)
