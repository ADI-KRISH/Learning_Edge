"""
Tutor Response Quality Evaluation Module
Measures: Semantic Similarity (BERTScore-style), ROUGE-L, Keyword Coverage, Faithfulness, Response Length.
Uses the existing BAAI/bge-small-en model for embeddings (no new model downloads).
"""
import numpy as np
from typing import List, Dict, Any


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """Computes ROUGE-L F1 score between reference and hypothesis."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "precision": round(scores["rougeL"].precision, 4),
            "recall": round(scores["rougeL"].recall, 4),
            "f1": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        print("  [WARN] rouge-score not installed. Run: uv add rouge-score")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _compute_semantic_similarity(reference: str, hypothesis: str, model) -> float:
    """Computes semantic similarity using sentence-transformers embeddings."""
    embeddings = model.encode([reference, hypothesis])
    return _cosine_similarity(embeddings[0], embeddings[1])


def _compute_keyword_coverage(response: str, keywords: List[str]) -> float:
    """Fraction of expected keywords that appear in the response."""
    if not keywords:
        return 0.0
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    return hits / len(keywords)


def _compute_faithfulness(response: str, rag_context: str, model) -> float:
    """
    Measures how grounded the response is in the retrieved context.
    Splits response into sentences, embeds each, and checks cosine similarity
    against the RAG context. Higher = more faithful/grounded.
    """
    if not rag_context or not response:
        return 0.0

    # Simple sentence splitting
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 15]
    if not sentences:
        return 0.0

    # Embed response sentences and RAG context chunks
    rag_chunks = [c.strip() for c in rag_context.split("\n\n") if len(c.strip()) > 20]
    if not rag_chunks:
        return 0.0

    sentence_embeddings = model.encode(sentences)
    chunk_embeddings = model.encode(rag_chunks)

    # For each response sentence, find max similarity to any RAG chunk
    sentence_scores = []
    for sent_emb in sentence_embeddings:
        max_sim = max(_cosine_similarity(sent_emb, chunk_emb) for chunk_emb in chunk_embeddings)
        sentence_scores.append(max_sim)

    return float(np.mean(sentence_scores))


def evaluate_responses(pipeline_results: List[Dict[str, Any]], gold_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes response quality metrics for all conceptual (non-quiz, non-follow-up) queries.
    Requires pipeline_results from run_evaluation with tutor_response and rag_context captured.
    """
    from sentence_transformers import SentenceTransformer
    from app.utils.config import EMBEDDING_MODEL

    print("\n🧠 Loading embedding model for response evaluation...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Build lookup from gold data
    gold_lookup = {g["id"]: g for g in gold_data}

    # Filter to conceptual queries with reference answers
    eval_results = [
        r for r in pipeline_results
        if gold_lookup.get(r["id"], {}).get("category") == "conceptual"
        and gold_lookup.get(r["id"], {}).get("reference_answer")
        and r.get("tutor_response")
    ]

    if not eval_results:
        print("  [WARN] No conceptual results to evaluate.")
        return {"num_evaluated": 0}

    rouge_scores = []
    semantic_scores = []
    keyword_scores = []
    faithfulness_scores = []
    lengths = []
    per_query = []

    for i, result in enumerate(eval_results):
        gold = gold_lookup[result["id"]]
        response = result["tutor_response"]
        reference = gold["reference_answer"]
        keywords = gold.get("relevant_doc_keywords", [])
        rag_context = result.get("rag_context", "")

        print(f"  [{i+1}/{len(eval_results)}] Evaluating response for: \"{gold['query']}\"")

        # ROUGE-L
        rouge = _compute_rouge_l(reference, response)
        rouge_scores.append(rouge["f1"])

        # Semantic Similarity
        sem_sim = _compute_semantic_similarity(reference, response, embed_model)
        semantic_scores.append(sem_sim)

        # Keyword Coverage
        kw_cov = _compute_keyword_coverage(response, keywords)
        keyword_scores.append(kw_cov)

        # Faithfulness
        faith = _compute_faithfulness(response, rag_context, embed_model)
        faithfulness_scores.append(faith)

        # Response Length (words)
        word_count = len(response.split())
        lengths.append(word_count)

        per_query.append({
            "id": result["id"],
            "query": gold["query"],
            "rouge_l_f1": rouge["f1"],
            "semantic_similarity": round(sem_sim, 4),
            "keyword_coverage": round(kw_cov, 4),
            "faithfulness": round(faith, 4),
            "word_count": word_count,
        })

    def _safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "num_evaluated": len(eval_results),
        "rouge_l_f1": _safe_mean(rouge_scores),
        "semantic_similarity": _safe_mean(semantic_scores),
        "keyword_coverage": _safe_mean(keyword_scores),
        "faithfulness": _safe_mean(faithfulness_scores),
        "avg_word_count": round(_safe_mean(lengths)),
        "per_query": per_query,
    }
