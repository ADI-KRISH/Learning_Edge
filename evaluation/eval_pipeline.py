"""
End-to-End Pipeline & Knowledge Graph Evaluation Module
Measures: Topic Detection Accuracy, Prerequisite Accuracy (Jaccard), Per-Node Latency, Total Latency, Peak RAM.
"""
from typing import List, Dict, Any


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0  # Both empty = perfect match
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def evaluate_pipeline(pipeline_results: List[Dict[str, Any]], gold_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates pipeline-level metrics using pre-captured results with timing data.
    
    Expected fields in each pipeline_result:
        - id, query
        - active_topic (detected by graph_analysis)
        - graph_context (prerequisites returned)
        - node_timings: dict of {node_name: seconds}
        - total_time: float seconds
        - peak_ram_mb: float (optional)
    """
    gold_lookup = {g["id"]: g for g in gold_data}

    # Filter to queries with expected topics (skip follow-ups with null expected_topic)
    eval_results = [
        r for r in pipeline_results
        if gold_lookup.get(r["id"], {}).get("expected_topic") is not None
    ]

    if not eval_results:
        print("  [WARN] No pipeline results to evaluate.")
        return {"num_evaluated": 0}

    topic_correct = 0
    jaccard_scores = []
    total_times = []
    node_timing_sums = {}  # aggregate per-node timings
    per_query = []

    for i, result in enumerate(eval_results):
        gold = gold_lookup[result["id"]]

        # --- Topic Detection Accuracy ---
        detected = result.get("active_topic", "")
        expected = gold["expected_topic"]
        is_correct = detected.lower().strip() == expected.lower().strip() if detected and expected else False
        if is_correct:
            topic_correct += 1

        # --- Prerequisite Accuracy (Jaccard) ---
        detected_prereqs = set(p.lower().strip() for p in result.get("graph_context", []))
        expected_prereqs = set(p.lower().strip() for p in gold.get("expected_prerequisites", []))
        jaccard = _jaccard_similarity(detected_prereqs, expected_prereqs)
        jaccard_scores.append(jaccard)

        # --- Latency ---
        total_time = result.get("total_time", 0.0)
        total_times.append(total_time)

        node_timings = result.get("node_timings", {})
        for node_name, elapsed in node_timings.items():
            if node_name not in node_timing_sums:
                node_timing_sums[node_name] = []
            node_timing_sums[node_name].append(elapsed)

        per_query.append({
            "id": result["id"],
            "query": gold["query"],
            "expected_topic": expected,
            "detected_topic": detected,
            "topic_correct": is_correct,
            "expected_prerequisites": gold.get("expected_prerequisites", []),
            "detected_prerequisites": result.get("graph_context", []),
            "jaccard_similarity": round(jaccard, 4),
            "total_time_sec": round(total_time, 2),
            "node_timings": {k: round(v, 3) for k, v in node_timings.items()},
        })

    n = len(eval_results)
    def _safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    # Compute average per-node latency
    avg_node_latency = {}
    for node_name, times in node_timing_sums.items():
        avg_node_latency[node_name] = round(sum(times) / len(times), 3)

    # Get peak RAM from results (if tracked)
    peak_ram = max((r.get("peak_ram_mb", 0) for r in pipeline_results), default=0)

    return {
        "num_evaluated": n,
        "topic_detection_accuracy": round(topic_correct / n, 4) if n else 0.0,
        "prerequisite_accuracy_jaccard": _safe_mean(jaccard_scores),
        "avg_total_latency_sec": round(_safe_mean(total_times), 2),
        "avg_node_latency_sec": avg_node_latency,
        "peak_ram_mb": round(peak_ram, 1),
        "per_query": per_query,
    }
