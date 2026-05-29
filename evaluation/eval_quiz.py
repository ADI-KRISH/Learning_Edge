"""
Quiz Generation Quality Evaluation Module
Measures: JSON Parse Rate, Schema Compliance, Question Count Accuracy, Topic Alignment, Answer Index Validity.
All deterministic checks — no LLM judge needed.
"""
import json
import re
from typing import List, Dict, Any


def _try_parse_quiz_json(raw_response: str) -> tuple:
    """
    Attempts to parse the quiz JSON from the LLM response.
    Returns (parsed_list_or_None, error_string_or_None).
    """
    if not raw_response or not raw_response.strip():
        return None, "Empty response"

    clean = raw_response.strip()
    # Strip markdown code fences
    clean = re.sub(r'^```json\s*', '', clean)
    clean = re.sub(r'^```\s*', '', clean)
    clean = re.sub(r'\s*```$', '', clean)

    # Find JSON array
    json_match = re.search(r'\[.*\]', clean, re.DOTALL)
    if json_match:
        clean = json_match.group(0)

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed, None
        else:
            return None, f"Parsed result is {type(parsed).__name__}, not list"
    except json.JSONDecodeError as e:
        return None, str(e)


def _validate_schema(questions: list) -> Dict[str, Any]:
    """Validates that each question has the required fields with correct types."""
    valid_count = 0
    issues = []

    for i, q in enumerate(questions):
        q_issues = []
        if not isinstance(q, dict):
            issues.append(f"Q{i+1}: not a dict")
            continue

        # Check required fields
        if "question" not in q or not isinstance(q.get("question"), str):
            q_issues.append("missing/invalid 'question'")
        if "options" not in q or not isinstance(q.get("options"), list):
            q_issues.append("missing/invalid 'options'")
        elif len(q["options"]) != 4:
            q_issues.append(f"options has {len(q['options'])} items (expected 4)")
        if "correct" not in q:
            q_issues.append("missing 'correct'")
        elif not isinstance(q["correct"], int) or q["correct"] < 0 or q["correct"] >= len(q.get("options", [0,1,2,3])):
            q_issues.append(f"invalid 'correct' index: {q.get('correct')}")
        if "explanation" not in q or not isinstance(q.get("explanation"), str):
            q_issues.append("missing/invalid 'explanation'")

        if not q_issues:
            valid_count += 1
        else:
            issues.append(f"Q{i+1}: {'; '.join(q_issues)}")

    return {
        "valid_count": valid_count,
        "total_count": len(questions),
        "compliance_rate": valid_count / len(questions) if questions else 0.0,
        "issues": issues,
    }


def _check_topic_alignment(questions: list, expected_topic: str, keywords: List[str]) -> float:
    """Checks if quiz questions are about the expected topic by keyword overlap."""
    if not questions or not expected_topic:
        return 0.0

    # Combine topic name and keywords for matching
    topic_terms = [expected_topic.lower()] + [kw.lower() for kw in keywords]

    aligned = 0
    for q in questions:
        q_text = q.get("question", "").lower()
        options_text = " ".join(q.get("options", [])).lower()
        full_text = f"{q_text} {options_text}"

        # A question is aligned if it contains at least 1 topic term
        if any(term in full_text for term in topic_terms):
            aligned += 1

    return aligned / len(questions) if questions else 0.0


def evaluate_quiz_generation(pipeline_results: List[Dict[str, Any]], gold_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates quiz generation quality for all quiz-type queries.
    Uses deterministic checks only (no LLM judge).
    """
    gold_lookup = {g["id"]: g for g in gold_data}

    # Filter to quiz requests
    quiz_results = [
        r for r in pipeline_results
        if gold_lookup.get(r["id"], {}).get("category") == "quiz_request"
    ]

    if not quiz_results:
        print("  [WARN] No quiz results to evaluate.")
        return {"num_evaluated": 0}

    parse_successes = 0
    schema_scores = []
    question_count_correct = 0
    topic_alignment_scores = []
    per_query = []

    for i, result in enumerate(quiz_results):
        gold = gold_lookup[result["id"]]
        raw_quiz = result.get("quiz_response", "")

        print(f"  [{i+1}/{len(quiz_results)}] Evaluating quiz for: \"{gold['query']}\"")

        # 1. JSON Parse
        parsed, parse_error = _try_parse_quiz_json(raw_quiz)
        is_parsed = parsed is not None
        if is_parsed:
            parse_successes += 1

        # 2. Schema Compliance
        schema_result = _validate_schema(parsed) if parsed else {"compliance_rate": 0.0, "issues": [parse_error or "parse failed"]}
        schema_scores.append(schema_result["compliance_rate"])

        # 3. Question Count (expected: 3)
        actual_count = len(parsed) if parsed else 0
        count_correct = actual_count == 3
        if count_correct:
            question_count_correct += 1

        # 4. Topic Alignment
        topic_align = 0.0
        if parsed:
            topic_align = _check_topic_alignment(
                parsed,
                gold.get("expected_topic", ""),
                gold.get("relevant_doc_keywords", [])
            )
        topic_alignment_scores.append(topic_align)

        per_query.append({
            "id": result["id"],
            "query": gold["query"],
            "json_parsed": is_parsed,
            "parse_error": parse_error,
            "question_count": actual_count,
            "schema_compliance": round(schema_result["compliance_rate"], 4),
            "schema_issues": schema_result.get("issues", []),
            "topic_alignment": round(topic_align, 4),
        })

    n = len(quiz_results)
    def _safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "num_evaluated": n,
        "json_parse_rate": round(parse_successes / n, 4) if n else 0.0,
        "schema_compliance": _safe_mean(schema_scores),
        "question_count_accuracy": round(question_count_correct / n, 4) if n else 0.0,
        "topic_alignment": _safe_mean(topic_alignment_scores),
        "per_query": per_query,
    }
