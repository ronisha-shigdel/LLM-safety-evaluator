import os
import time
from datetime import datetime

from groq import Groq

from test_cases import TEST_CASES
from evaluator import evaluate
from reporter import save_to_csv, print_summary

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

RESPONSE_MODEL = "llama-3.3-70b-versatile"

results = []

print("=" * 60)
print("       AI QA EVALUATION FRAMEWORK")
print(f"       Running {len(TEST_CASES)} test cases...")
print("=" * 60)

for test in TEST_CASES:
    print(f"\n[{test['id']}] {test['category']} (risk: {test['risk_level']})")
    print(f"  Prompt: {test['prompt'][:80]}...")

    ai_response = groq_client.chat.completions.create(
        model=RESPONSE_MODEL,
        messages=[{"role": "user", "content": test["prompt"]}],
        temperature=0.7,
    ).choices[0].message.content

    evaluation = evaluate(groq_client, test, ai_response)

    result = {
        "test_id": test["id"],
        "category": test["category"],
        "risk_level": test["risk_level"],
        "prompt": test["prompt"],
        "ai_response": ai_response,
        "safety_score": evaluation.get("safety_score", 0),
        "fairness_score": evaluation.get("fairness_score", 0),
        "helpfulness_score": evaluation.get("helpfulness_score", 0),
        "clarity_score": evaluation.get("clarity_score", 0),
        "consistency_score": evaluation.get("consistency_score", 0),
        "overall_score": evaluation.get("overall_score", 0),
        "pass_fail": evaluation.get("pass_fail", "FAIL"),
        "flags": evaluation.get("flags", "none"),
        "summary": evaluation.get("summary", ""),
        "timestamp": datetime.now().isoformat(),
    }

    results.append(result)

    status = "✅ PASS" if result["pass_fail"] == "PASS" else "❌ FAIL"
    print(f"  Result: {status}  |  Overall: {result['overall_score']}/10  |  Safety: {result['safety_score']}  |  Fairness: {result['fairness_score']}")

    time.sleep(0.5)

csv_path = save_to_csv(results)
print(f"\n📄 Results saved to: {csv_path}")

print_summary(results)
