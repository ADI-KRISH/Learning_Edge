"""
Master Evaluation Runner for the Offline AI Tutor
Runs all evaluation modules, aggregates results, and generates a report.

Usage:
    python -m evaluation.run_evaluation
"""
import json
import os
import sys
import time
import io
from datetime import datetime
from typing import Dict, Any, List


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_DATASET_PATH = os.path.join(EVAL_DIR, "gold_dataset.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def load_gold_dataset() -> List[Dict[str, Any]]:
    """Loads the gold standard test dataset."""
    with open(GOLD_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_for_query(query: str, query_id: str, user_id: str = "eval_user") -> Dict[str, Any]:
    """
    Runs the full LangGraph pipeline for a single query and captures
    all intermediate state + per-node timing.
    """
    from app.agents.orchestrator import stream_tutor_pipeline

    result = {
        "id": query_id,
        "query": query,
        "active_topic": "",
        "graph_context": [],
        "rag_context": "",
        "tutor_response": "",
        "quiz_response": "",
        "needs_quiz": False,
        "node_timings": {},
        "total_time": 0.0,
        "peak_ram_mb": 0.0,
    }

    # Track RAM if psutil available
    try:
        import psutil
        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        process = None
        ram_before = 0

    start = time.time()
    prev_time = start

    try:
        for node_name, state in stream_tutor_pipeline(query, user_id):
            now = time.time()
            result["node_timings"][node_name] = round(now - prev_time, 4)
            prev_time = now

            # Capture state from each node (state is a dict from LangGraph)
            if isinstance(state, dict):
                if state.get("active_topic"):
                    result["active_topic"] = state["active_topic"]
                if state.get("graph_context"):
                    result["graph_context"] = state["graph_context"]
                if state.get("rag_context"):
                    result["rag_context"] = state["rag_context"]
                if state.get("tutor_response"):
                    result["tutor_response"] = state["tutor_response"]
                if state.get("quiz_response"):
                    result["quiz_response"] = state["quiz_response"]
                if state.get("needs_quiz"):
                    result["needs_quiz"] = state["needs_quiz"]
    except Exception as e:
        result["error"] = str(e)

    result["total_time"] = round(time.time() - start, 4)

    # Peak RAM
    if process:
        ram_after = process.memory_info().rss / (1024 * 1024)
        result["peak_ram_mb"] = round(max(ram_before, ram_after), 1)

    return result


def run_all_pipelines(gold_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs the pipeline for every gold dataset query and captures results."""
    results = []
    # Skip follow-up queries for pipeline execution (they need prior context)
    executable = [g for g in gold_data if g.get("category") != "follow_up"]

    print(f"\n🚀 Running pipeline for {len(executable)} queries (this will take several minutes)...\n")

    for i, entry in enumerate(executable):
        query = entry["query"]
        print(f"  [{i+1}/{len(executable)}] Pipeline: \"{query}\"")

        result = run_pipeline_for_query(query, entry["id"])

        if result.get("error"):
            print(f"    ⚠️  Error: {result['error']}")
        else:
            topic = result.get("active_topic", "—")
            response_len = len(result.get("tutor_response", "").split())
            elapsed = result.get("total_time", 0)
            print(f"    ✓ Topic: {topic} | Response: {response_len} words | Time: {elapsed:.1f}s")

        results.append(result)

    return results


def print_report(
    retrieval_metrics: Dict,
    response_metrics: Dict,
    quiz_metrics: Dict,
    pipeline_metrics: Dict,
):
    """Prints a formatted evaluation report to the terminal."""
    print("\n")
    print("=" * 65)
    print("         OFFLINE AI TUTOR -- EVALUATION REPORT")
    print(f"         Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # --- RAG Retrieval ---
    print("\n[RAG] RETRIEVAL QUALITY")
    print("-" * 40)
    h = retrieval_metrics.get("hybrid", {})
    v = retrieval_metrics.get("vector_only", {})
    b = retrieval_metrics.get("bm25_only", {})
    print(f"  Queries Evaluated:   {retrieval_metrics.get('num_queries_evaluated', 0)}")
    print(f"  {'Metric':<22} {'Hybrid':>8}  {'Vector':>8}  {'BM25':>8}")
    print(f"  {'-'*22} {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Keyword Hit Rate':<22} {h.get('keyword_hit_rate', 0):>7.1%}  {v.get('keyword_hit_rate', 0):>7.1%}  {b.get('keyword_hit_rate', 0) if b else 'N/A':>8}")
    print(f"  {'Precision@K':<22} {h.get('precision_at_k', 0):>7.1%}  {v.get('precision_at_k', 0):>7.1%}  {b.get('precision_at_k', 0) if b else 'N/A':>8}")
    print(f"  {'MRR':<22} {h.get('mrr', 0):>8.4f}  {v.get('mrr', 0):>8.4f}  {b.get('mrr', 0) if b else 'N/A':>8}")

    # --- Tutor Response ---
    print("\n[LLM] TUTOR RESPONSE QUALITY")
    print("-" * 40)
    print(f"  Queries Evaluated:   {response_metrics.get('num_evaluated', 0)}")
    print(f"  Semantic Similarity: {response_metrics.get('semantic_similarity', 0):.4f}")
    print(f"  ROUGE-L (F1):        {response_metrics.get('rouge_l_f1', 0):.4f}")
    print(f"  Keyword Coverage:    {response_metrics.get('keyword_coverage', 0):.1%}")
    print(f"  Faithfulness:        {response_metrics.get('faithfulness', 0):.4f}")
    print(f"  Avg Response Length:  {response_metrics.get('avg_word_count', 0)} words")

    # --- Quiz Generation ---
    print("\n[QUIZ] QUIZ GENERATION QUALITY")
    print("-" * 40)
    print(f"  Queries Evaluated:      {quiz_metrics.get('num_evaluated', 0)}")
    print(f"  JSON Parse Rate:        {quiz_metrics.get('json_parse_rate', 0):.1%}")
    print(f"  Schema Compliance:      {quiz_metrics.get('schema_compliance', 0):.1%}")
    print(f"  Question Count Acc:     {quiz_metrics.get('question_count_accuracy', 0):.1%}")
    print(f"  Topic Alignment:        {quiz_metrics.get('topic_alignment', 0):.1%}")

    # --- Pipeline & Knowledge Graph ---
    print("\n[KG] KNOWLEDGE GRAPH & PIPELINE")
    print("-" * 40)
    print(f"  Queries Evaluated:       {pipeline_metrics.get('num_evaluated', 0)}")
    print(f"  Topic Detection Acc:     {pipeline_metrics.get('topic_detection_accuracy', 0):.1%}")
    print(f"  Prerequisite Acc (Jac):  {pipeline_metrics.get('prerequisite_accuracy_jaccard', 0):.4f}")
    print(f"  Avg Latency/Query:       {pipeline_metrics.get('avg_total_latency_sec', 0):.1f}s")
    if pipeline_metrics.get("avg_node_latency_sec"):
        for node, t in pipeline_metrics["avg_node_latency_sec"].items():
            print(f"    └─ {node:<12}  {t:.3f}s")
    if pipeline_metrics.get("peak_ram_mb"):
        print(f"  Peak RAM:                {pipeline_metrics['peak_ram_mb']:.0f} MB")

    print("\n" + "=" * 65)


def save_report(
    retrieval_metrics: Dict,
    response_metrics: Dict,
    quiz_metrics: Dict,
    pipeline_metrics: Dict,
):
    """Saves the full evaluation report as JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESULTS_DIR, f"eval_report_{timestamp}.json")

    report = {
        "timestamp": datetime.now().isoformat(),
        "retrieval": retrieval_metrics,
        "response_quality": response_metrics,
        "quiz_generation": quiz_metrics,
        "pipeline": pipeline_metrics,
    }

    # Remove per-query details from the summary (keep them in a separate key)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 Report saved to: {report_path}")
    return report_path


def main():
    """Main evaluation entry point."""
    # Force UTF-8 output on Windows to avoid cp1252 crashes
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("=" * 59)
    print("       OFFLINE AI TUTOR -- EVALUATION SUITE")
    print("=" * 59)

    # 1. Load gold dataset
    print("\n📋 Loading gold dataset...")
    gold_data = load_gold_dataset()
    print(f"   Loaded {len(gold_data)} test cases")

    # 2. Run RAG Retrieval Evaluation (no LLM needed)
    print("\n" + "=" * 55)
    print("  PHASE 1: RAG Retrieval Evaluation (no LLM)")
    print("=" * 55)
    from evaluation.eval_retrieval import evaluate_retrieval
    retrieval_metrics = evaluate_retrieval(gold_data)
    print(f"   ✓ Retrieval evaluation complete")

    # 3. Run full pipeline for all queries (requires Ollama)
    print("\n" + "=" * 55)
    print("  PHASE 2: Full Pipeline Execution (requires Ollama)")
    print("=" * 55)
    try:
        pipeline_results = run_all_pipelines(gold_data)
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        print("   Make sure Ollama is running: `ollama serve`")
        pipeline_results = []

    # 4. Evaluate Response Quality
    print("\n" + "=" * 55)
    print("  PHASE 3: Response Quality Evaluation")
    print("=" * 55)
    if pipeline_results:
        from evaluation.eval_response import evaluate_responses
        response_metrics = evaluate_responses(pipeline_results, gold_data)
    else:
        response_metrics = {"num_evaluated": 0, "error": "Pipeline did not run"}

    # 5. Evaluate Quiz Generation
    print("\n" + "=" * 55)
    print("  PHASE 4: Quiz Generation Evaluation")
    print("=" * 55)
    if pipeline_results:
        from evaluation.eval_quiz import evaluate_quiz_generation
        quiz_metrics = evaluate_quiz_generation(pipeline_results, gold_data)
    else:
        quiz_metrics = {"num_evaluated": 0, "error": "Pipeline did not run"}

    # 6. Evaluate Pipeline & Knowledge Graph
    print("\n" + "=" * 55)
    print("  PHASE 5: Pipeline & Knowledge Graph Evaluation")
    print("=" * 55)
    if pipeline_results:
        from evaluation.eval_pipeline import evaluate_pipeline
        pipeline_metrics = evaluate_pipeline(pipeline_results, gold_data)
    else:
        pipeline_metrics = {"num_evaluated": 0, "error": "Pipeline did not run"}

    # 7. Print Report
    print_report(retrieval_metrics, response_metrics, quiz_metrics, pipeline_metrics)

    # 8. Save JSON Report
    report_path = save_report(retrieval_metrics, response_metrics, quiz_metrics, pipeline_metrics)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
