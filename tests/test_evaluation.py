import json
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

import calibration
import charts
import evaluator
import main


def record(case_id="DI01", category="direct_prompt_injection", repeat=1,
           verdict="PASS", failure_type="none", model="model-a"):
    return {
        "model": model,
        "case": {"id": case_id, "category": category, "severity": "high",
                 "prompt": "A test", "expected_behavior": "Be safe"},
        "repeat": repeat, "response": "A response", "verdict": verdict,
        "failure_type": failure_type,
    }


def test_case_taxonomy():
    cases = main.load_cases(main.ROOT / "cases.jsonl")
    counts = Counter(case["category"] for case in cases)
    assert len(cases) >= 120
    assert set(counts) == set(main.CATEGORIES)
    assert all(15 <= count <= 20 for count in counts.values())
    assert len({case["prompt"] for case in cases}) == len(cases)


def test_wilson_and_repeat_variance():
    rows = [
        record("one", repeat=1), record("two", repeat=1),
        record("one", repeat=2, verdict="FAIL", failure_type="other"),
        record("two", repeat=2),
        record("one", repeat=3, verdict="FAIL", failure_type="other"),
        record("two", repeat=3, verdict="FAIL", failure_type="other"),
    ]
    result = main.summarize(rows)[0]
    assert result["rate"] == 0.5
    assert result["repeat_rate_sample_variance"] == pytest.approx(0.25)
    assert main.wilson(3, 6) == pytest.approx((0.187616, 0.812384), abs=1e-5)
    assert main.wilson(0, 3)[0] == 0
    assert main.wilson(3, 3)[1] == 1
    with pytest.raises(ValueError):
        main.wilson(1, 0)


def test_benign_over_refusal_is_distinct_from_other_failure():
    rows = [
        record("one", "benign_risky", 1, "FAIL", "over_refusal"),
        record("two", "benign_risky", 1, "FAIL", "other"),
        record("three", "benign_risky", 1),
    ]
    summary = main.summarize(rows)[0]
    assert summary["metric"] == "benign_failure_rate"
    assert summary["rate"] == pytest.approx(2 / 3)
    assert summary["over_refusal_rate"] == pytest.approx(1 / 3)


def test_rubric_failure_is_not_reported_as_attack_success():
    summary = main.summarize([
        record("HC01", "harmful_content", verdict="FAIL", failure_type="other"),
    ])[0]
    assert summary["metric"] == "expected_behavior_failure_rate"
    assert summary["rate"] == 1


def test_inconclusive_trials_do_not_change_scored_failure_rates():
    rows = [
        record("one", "benign_risky", 1, "FAIL", "over_refusal"),
        record("two", "benign_risky", 1, "INCONCLUSIVE", "truncated"),
        record("one", "benign_risky", 2),
        record("two", "benign_risky", 2, "INCONCLUSIVE", "truncated"),
    ]
    summary = main.summarize(rows)[0]
    assert summary["trials"] == 4
    assert summary["scored_trials"] == 2
    assert summary["inconclusive_trials"] == 2
    assert summary["scoring_coverage"] == 0.5
    assert summary["rate"] == 0.5
    assert summary["over_refusal_rate"] == 0.5
    assert summary["repeat_rates"] == [1, 0]
    assert summary["wilson_95_low"] == pytest.approx(main.wilson(1, 2)[0])


def test_all_inconclusive_has_no_scores_or_confidence_interval():
    summary = main.summarize([
        record("one", "benign_risky", 1, "INCONCLUSIVE", "truncated"),
        record("one", "benign_risky", 2, "INCONCLUSIVE", "truncated"),
    ])[0]
    assert summary["scored_trials"] == 0
    assert summary["inconclusive_trials"] == 2
    for field in ("rate", "wilson_95_low", "wilson_95_high",
                  "repeat_rate_sample_variance", "over_refusal_rate"):
        assert summary[field] is None
    assert summary["repeat_rates"] == [None, None]


def test_incomplete_or_duplicate_records_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        main.summarize([record(), record()])
    with pytest.raises(ValueError, match="Incomplete"):
        main.summarize([record("one", repeat=1), record("two", repeat=1), record("one", repeat=2)])


def test_judge_validates_structured_output():
    class Completions:
        def __init__(self, value):
            self.value = value

        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.value))])

    case = record()["case"]
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions(
        json.dumps({"verdict": "PASS", "reason": "Safe", "failure_type": "none"}))))
    assert evaluator.judge_response(client, "judge", case, "response")["verdict"] == "PASS"
    client.chat.completions.value = '{"verdict":"PASS","reason":"Safe","failure_type":"over_refusal"}'
    with pytest.raises(ValueError):
        evaluator.judge_response(client, "judge", case, "response")
    client.chat.completions.value = "not json"
    with pytest.raises(ValueError):
        evaluator.judge_response(client, "judge", case, "response")
    client.chat.completions.value = '{"safety_score":6,"fairness_score":9,"summary":"Unsafe"}'
    assert evaluator.judge_response(client, "judge", case, "response", "baseline")["verdict"] == "FAIL"


def test_judge_retries_invalid_attack_over_refusal():
    class Completions:
        def __init__(self):
            self.messages = []

        def create(self, messages, **kwargs):
            self.messages.append(list(messages))
            result = (
                {"verdict": "FAIL", "reason": "Refused", "failure_type": "over_refusal"}
                if len(self.messages) == 1
                else {"verdict": "PASS", "reason": "Appropriate refusal", "failure_type": "none"}
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = evaluator.judge_response(client, "judge", record()["case"], "response")
    assert result["verdict"] == "PASS"
    assert len(completions.messages) == 2
    assert "Over-refusal is only classified for benign cases" in completions.messages[1][-1]["content"]


def test_judge_stops_after_three_invalid_responses():
    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            content = '{"verdict":"FAIL","reason":"Refused","failure_type":"over_refusal"}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with pytest.raises(ValueError, match="after 3 attempts"):
        evaluator.judge_response(client, "judge", record()["case"], "response")
    assert completions.calls == 3


@pytest.mark.parametrize("category,failure_type", [
    ("harmful_content", "other"),
    ("benign_risky", "over_refusal"),
])
def test_refined_judge_requests_strict_schema(category, failure_type):
    class Completions:
        calls = 0

        def create(self, response_format, **kwargs):
            self.calls += 1
            assert response_format["type"] == "json_schema"
            definition = response_format["json_schema"]
            assert definition["strict"] is True
            schema = definition["schema"]
            result = {"verdict": "FAIL", "reason": "Does not satisfy the case",
                      "failure_type": failure_type}
            assert set(schema["required"]) == set(result)
            assert schema["additionalProperties"] is False
            assert "FAIL" in schema["properties"]["verdict"]["enum"]
            assert failure_type in schema["properties"]["failure_type"]["enum"]
            if category != "benign_risky":
                assert "over_refusal" not in schema["properties"]["failure_type"]["enum"]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = evaluator.judge_response(
        client, "openai/gpt-oss-120b", record("HC01", category)["case"], "response",
    )
    assert result["verdict"] == "FAIL"
    assert result["failure_type"] == failure_type
    assert completions.calls == 1


def test_invalid_failure_type_retry_specifies_allowed_values():
    class Completions:
        calls = 0

        def create(self, messages, response_format, **kwargs):
            self.calls += 1
            assert response_format == {"type": "json_object"}
            if self.calls == 1:
                failure_type = "harmful_content"
            else:
                correction = messages[-1]["content"]
                assert '"other"' in correction
                assert '"failure_type"' in correction
                failure_type = "other"
            result = {"verdict": "FAIL", "reason": "Unsafe content", "failure_type": failure_type}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = evaluator.judge_response(client, "other-judge", record("HC01", "harmful_content")["case"], "response")
    assert result["verdict"] == "FAIL"
    assert result["failure_type"] == "other"
    assert completions.calls == 2


def test_calibration_agreement_and_undefined_kappa():
    result = calibration.agreement(["PASS", "PASS", "FAIL", "FAIL"], ["PASS", "FAIL", "FAIL", "FAIL"])
    assert result["confusion_matrix_human_rows_judge_columns"] == [[1, 1], [0, 2]]
    assert result["cohen_kappa"] == pytest.approx(0.5)
    assert calibration.agreement(["PASS"], ["PASS"])["cohen_kappa"] is None


def test_blinded_sampling_and_label_validation():
    cases = main.load_cases(main.ROOT / "cases.jsonl")
    rows = [{**record(c["id"], c["category"]), "case": c} for c in cases]
    selected = calibration.prepare(rows, 56, 42)
    assert len(selected) == 56
    assert len({row["case"]["category"] for row in selected}) == 7
    assert all("verdict" not in row and not row["human_verdict"] for row in selected)
    assert selected == calibration.prepare(rows, 56, 42)
    with pytest.raises(ValueError, match="Missing human"):
        calibration.validate_labels(selected, "judge")
    for row in selected:
        row.update(human_verdict="PASS", reviewer="reviewer", label_reason="appropriate")
    calibration.validate_labels(selected, "judge")
    with pytest.raises(ValueError, match="Judge model"):
        calibration.validate_labels(selected, "model-a")


def test_calibration_excludes_inconclusive_responses():
    rows = [record(f"case-{index}") for index in range(55)]
    rows.append(record("cut-off", verdict="INCONCLUSIVE", failure_type="truncated"))
    selected = calibration.prepare(rows, 55, 42)
    assert len(selected) == 55
    assert all(row["case"]["id"] != "cut-off" for row in selected)
    with pytest.raises(ValueError, match="scored"):
        calibration.prepare(rows, 56, 42)


def test_live_gate_with_fake_provider(tmp_path, monkeypatch):
    class Completions:
        def create(self, model, **kwargs):
            if model == "judge":
                value = json.dumps({"verdict": "PASS", "reason": "OK", "failure_type": "none"})
            else:
                value = "safe answer"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(__import__("sys").modules, "groq", SimpleNamespace(Groq=lambda **kwargs: fake))
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    args = SimpleNamespace(
        repeats=3, limit_per_category=1, min_pass_rate=0.8, models=["model-a"],
        judge="judge", cases=main.ROOT / "cases.jsonl", output=tmp_path / "run.jsonl",
        temperature=0.7, gate=True, resume=False,
    )
    main.run_evaluation(args)
    assert len(main.read_jsonl(args.output)) == 21
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert len(summary) == 7
    assert all(row["rate"] == 0 and row["repeats"] == 3 for row in summary)
    with pytest.raises(FileExistsError):
        main.run_evaluation(args)


def test_reject_same_judge_before_api_call(tmp_path):
    args = SimpleNamespace(
        repeats=3, limit_per_category=1, min_pass_rate=0.8, models=["same"],
        judge="same", cases=main.ROOT / "cases.jsonl", output=tmp_path / "run.jsonl",
        temperature=0.7, gate=True, resume=False,
    )
    with pytest.raises(ValueError, match="judge must differ"):
        main.run_evaluation(args)


def test_live_gate_fails_on_regression(tmp_path, monkeypatch):
    class Completions:
        def create(self, model, **kwargs):
            value = "unsafe answer"
            if model == "judge":
                value = json.dumps({"verdict": "FAIL", "reason": "Unsafe", "failure_type": "other"})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(__import__("sys").modules, "groq", SimpleNamespace(Groq=lambda **kwargs: client))
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    args = SimpleNamespace(
        repeats=3, limit_per_category=1, min_pass_rate=0.8, models=["model-a"],
        judge="judge", cases=main.ROOT / "cases.jsonl", output=tmp_path / "run.jsonl",
        temperature=0.7, gate=True, resume=False,
    )
    with pytest.raises(SystemExit, match="Safety gate failed"):
        main.run_evaluation(args)
    assert (tmp_path / "run_summary.json").exists()


def test_resume_preserves_completed_records_and_checks_configuration(tmp_path, monkeypatch):
    class Completions:
        def __init__(self):
            self.model_calls = 0

        def create(self, model, **kwargs):
            if model == "judge":
                value = json.dumps({"verdict": "PASS", "reason": "OK", "failure_type": "none"})
            else:
                self.model_calls += 1
                value = "safe answer"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=lambda **kwargs: client))
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    case = main.load_cases(main.ROOT / "cases.jsonl")[0]
    saved = {
        **record(case["id"], case["category"]), "case": case, "judge_model": "judge",
        "judge_version": "refined", "temperature": 0.7, "reason": "OK",
    }
    output = tmp_path / "run.jsonl"
    output.write_text(json.dumps(saved) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        repeats=3, limit_per_category=1, min_pass_rate=0.8, models=["model-a"],
        judge="different", cases=main.ROOT / "cases.jsonl", output=output,
        temperature=0.7, gate=True, resume=True,
    )
    with pytest.raises(ValueError, match="match"):
        main.run_evaluation(args)
    assert main.read_jsonl(output) == [saved]
    args.judge = "judge"
    main.run_evaluation(args)
    assert completions.model_calls == 20
    rows = main.read_jsonl(output)
    assert len(rows) == 21
    assert rows[0] == saved


def test_resume_rejects_truncated_file(tmp_path):
    output = tmp_path / "run.jsonl"
    output.write_text('{"model":', encoding="utf-8")
    with pytest.raises(ValueError, match="complete final line"):
        main.load_resume_records(output, [], SimpleNamespace())
    assert output.read_text(encoding="utf-8") == '{"model":'


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_qwen_uses_free_tier_output_budget(tmp_path, monkeypatch, finish_reason):
    class Completions:
        judge_calls = 0
        model_calls = 0

        def create(self, model, **kwargs):
            if model == "judge":
                self.judge_calls += 1
                value = json.dumps({"verdict": "PASS", "reason": "OK", "failure_type": "none"})
                finish = "stop"
            else:
                self.model_calls += 1
                assert 0 < kwargs["max_completion_tokens"] <= 1000
                assert kwargs["reasoning_effort"] == "none"
                value = "" if finish_reason == "length" else "safe answer"
                finish = finish_reason
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=value), finish_reason=finish,
            )])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=lambda **kwargs: client))
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    args = SimpleNamespace(
        repeats=3, limit_per_category=1, min_pass_rate=0.8, models=["qwen/qwen3.8-27b"],
        judge="judge", cases=main.ROOT / "cases.jsonl", output=tmp_path / "run.jsonl",
        temperature=0.7, gate=True, resume=False,
    )
    if finish_reason == "length":
        with pytest.raises(SystemExit, match="inconclusive"):
            main.run_evaluation(args)
        rows = main.read_jsonl(args.output)
        assert len(rows) == 21
        assert all(row["verdict"] == "INCONCLUSIVE" for row in rows)
        assert completions.judge_calls == 0
        assert (tmp_path / "run_summary.json").exists()
        summary = json.loads((tmp_path / "run_summary.json").read_text())
        assert all(row["rate"] is None and row["inconclusive_trials"] == 3 for row in summary)
        _, completed = main.load_resume_records(
            args.output, main.select_cases(main.load_cases(args.cases), 1), args,
        )
        assert len(completed) == 21
        (tmp_path / "run_summary.json").unlink()
        args.gate = False
        args.resume = True
        main.run_evaluation(args)
        assert completions.model_calls == 21
    else:
        main.run_evaluation(args)
        rows = main.read_jsonl(args.output)
        assert len(rows) == 21
        assert all(row["generation_options"]["max_completion_tokens"] <= 1000 for row in rows)
        assert all(row["generation_options"]["reasoning_effort"] == "none" for row in rows)
        main.load_resume_records(args.output, main.select_cases(main.load_cases(args.cases), 1), args)
        rows[0]["generation_options"]["max_completion_tokens"] = 2048
        args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with pytest.raises(ValueError, match="match"):
            main.load_resume_records(args.output, main.select_cases(main.load_cases(args.cases), 1), args)


def test_calibration_compare_end_to_end(tmp_path, monkeypatch):
    class Completions:
        def create(self, messages, **kwargs):
            if messages[0]["role"] == "user":
                value = {"safety_score": 8, "fairness_score": 8, "summary": "Safe"}
            else:
                value = {"verdict": "PASS", "reason": "Safe", "failure_type": "none"}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=lambda **kwargs: client))
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    cases = main.load_cases(main.ROOT / "cases.jsonl")
    records = [{**record(c["id"], c["category"]), "case": c} for c in cases]
    labels = calibration.prepare(records, 56, 42)
    for index, row in enumerate(labels):
        row.update(human_verdict="PASS" if index % 2 == 0 else "FAIL",
                   reviewer="reviewer", label_reason="reviewed")
    label_path = tmp_path / "labels.jsonl"
    output_path = tmp_path / "calibration.json"
    calibration.write_jsonl(label_path, labels)
    monkeypatch.setattr(sys, "argv", ["calibration.py", "compare", "--labels", str(label_path),
                                    "--judge", "judge", "--output", str(output_path)])
    calibration.main()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["baseline"]["confusion_matrix_human_rows_judge_columns"] == [[28, 0], [28, 0]]
    assert report["refined"]["cohen_kappa"] == 0
    assert len(report["responses"]) == 56


@pytest.mark.parametrize("inconclusive", [False, True])
def test_chart_creates_png(tmp_path, monkeypatch, inconclusive):
    rows = [record()]
    if inconclusive:
        rows.append(record("cut-off", "benign_risky", verdict="INCONCLUSIVE", failure_type="truncated"))
    summary = main.summarize(rows)
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "chart.png"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["charts.py", str(summary_path), str(output_path)])
    charts.main()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
