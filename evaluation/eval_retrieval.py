"""
RAG Retrieval Evaluation Module
Measures: Keyword Hit Rate, Precision@K, MRR, and Hybrid vs Vector-only vs BM25-only ablation.
"""
import time
from typing import List, Dict, Any


def _chunk_is_relevant(chunk_text: str, keywords: List[str], min_matches: int = 2) -> bool:
    """A chunk is considered relevant if it contains at least `min_matches` of the expected keywords."""
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches >= min_matches


def _compute_metrics(retrieved_nodes: list, keywords: List[str], k: int = 4) -> Dict[str, float]:
    """Computes Precision@K, MRR, and Keyword Hit Rate for a single query."""
    if not keywords:
        return {"precision_at_k": 0.0, "mrr": 0.0, "keyword_hit_rate": 0.0}

    relevant_count = 0
    first_relevant_rank = None
    keyword_hits = set()

    for i, node in enumerate(retrieved_nodes[:k]):
        text = node.get_text() if hasattr(node, "get_text") else str(node)
        text_lower = text.lower()

        is_relevant = _chunk_is_relevant(text, keywords)
        if is_relevant:
            relevant_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = i + 1  # 1-indexed

        for kw in keywords:
            if kw.lower() in text_lower:
                keyword_hits.add(kw.lower())

    precision_at_k = relevant_count / k if k > 0 else 0.0
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
    keyword_hit_rate = len(keyword_hits) / len(keywords) if keywords else 0.0

    return {
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "keyword_hit_rate": keyword_hit_rate,
    }


def evaluate_retrieval(gold_data: List[Dict[str, Any]], num_chunks: int = 4) -> Dict[str, Any]:
    """
    Runs retrieval for each gold query and computes aggregate metrics.
    Also runs vector-only and BM25-only ablation for comparison.
    """
    from app.rag.retrieval import HybridRAGRetriever
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    print("\n📚 Initializing Retriever for evaluation...")
    retriever = HybridRAGRetriever()

    # Pre-build BM25 and vector-only retrievers for ablation
    vector_retriever = retriever.index.as_retriever(similarity_top_k=num_chunks)
    try:
        chroma_data = retriever.chroma_collection.get()
        text_nodes = []
        if chroma_data and "ids" in chroma_data and chroma_data["ids"]:
            for doc_id, text, metadata in zip(
                chroma_data["ids"], chroma_data["documents"], chroma_data["metadatas"]
            ):
                text_nodes.append(TextNode(id_=doc_id, text=text, metadata=metadata or {}))

        bm25_retriever = BM25Retriever.from_defaults(nodes=text_nodes, similarity_top_k=num_chunks)
        bm25_available = True
    except Exception as e:
        print(f"  [WARN] BM25 ablation unavailable: {e}")
        bm25_available = False

    # Filter to queries that have keywords to evaluate
    eval_queries = [q for q in gold_data if q.get("relevant_doc_keywords")]

    hybrid_metrics_list = []
    vector_metrics_list = []
    bm25_metrics_list = []
    per_query_results = []

    for i, entry in enumerate(eval_queries):
        query = entry["query"]
        keywords = entry["relevant_doc_keywords"]
        print(f"  [{i+1}/{len(eval_queries)}] Retrieving for: \"{query}\"")

        # --- Hybrid Retrieval ---
        try:
            hybrid_ret = retriever.get_retriever(num_chunks=num_chunks)
            hybrid_nodes = hybrid_ret.retrieve(query)
        except Exception:
            hybrid_nodes = []
        hybrid_m = _compute_metrics(hybrid_nodes, keywords, k=num_chunks)
        hybrid_metrics_list.append(hybrid_m)

        # --- Vector-only Retrieval ---
        try:
            vector_nodes = vector_retriever.retrieve(query)
        except Exception:
            vector_nodes = []
        vector_m = _compute_metrics(vector_nodes, keywords, k=num_chunks)
        vector_metrics_list.append(vector_m)

        # --- BM25-only Retrieval ---
        if bm25_available:
            try:
                bm25_nodes = bm25_retriever.retrieve(query)
            except Exception:
                bm25_nodes = []
            bm25_m = _compute_metrics(bm25_nodes, keywords, k=num_chunks)
            bm25_metrics_list.append(bm25_m)

        per_query_results.append({
            "id": entry["id"],
            "query": query,
            "hybrid": hybrid_m,
            "vector_only": vector_m,
            "bm25_only": bm25_m if bm25_available else None,
        })

    # Aggregate averages
    def _avg(metrics_list, key):
        vals = [m[key] for m in metrics_list if m]
        return sum(vals) / len(vals) if vals else 0.0

    results = {
        "num_queries_evaluated": len(eval_queries),
        "hybrid": {
            "keyword_hit_rate": round(_avg(hybrid_metrics_list, "keyword_hit_rate"), 4),
            "precision_at_k": round(_avg(hybrid_metrics_list, "precision_at_k"), 4),
            "mrr": round(_avg(hybrid_metrics_list, "mrr"), 4),
        },
        "vector_only": {
            "keyword_hit_rate": round(_avg(vector_metrics_list, "keyword_hit_rate"), 4),
            "precision_at_k": round(_avg(vector_metrics_list, "precision_at_k"), 4),
            "mrr": round(_avg(vector_metrics_list, "mrr"), 4),
        },
        "bm25_only": {
            "keyword_hit_rate": round(_avg(bm25_metrics_list, "keyword_hit_rate"), 4),
            "precision_at_k": round(_avg(bm25_metrics_list, "precision_at_k"), 4),
            "mrr": round(_avg(bm25_metrics_list, "mrr"), 4),
        } if bm25_available else None,
        "per_query": per_query_results,
    }

    return results
