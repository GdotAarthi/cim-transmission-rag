"""
CIM RAG Evaluation
Measures retrieval quality (hit rate, MRR) and answer faithfulness.
Run this after building the vector store to benchmark your RAG pipeline.

Metrics computed:
  - Hit Rate @ k: % of queries where the correct CIM object is in top-k results
  - MRR (Mean Reciprocal Rank): average of 1/rank for first correct result
  - Answer faithfulness: manual spot-check set with expected answer fragments

Usage:
    python notebooks/evaluate_rag.py
"""

import sys
import os
import json
from dataclasses import dataclass, asdict
from typing import List
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.embeddings.vector_store import get_collection, similarity_search
from src.retrieval.rag_chain import CIMRagChain


# ---------------------------------------------------------------------------
# Ground truth evaluation set — extend this as you add more CIM data
# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    question: str
    expected_object_ids: List[str]   # correct CIM object IDs that MUST appear
    expected_fragments: List[str]    # strings that should appear in the answer


EVAL_SET: List[EvalCase] = [
    EvalCase(
        question="What is the positive-sequence reactance of the Charlotte-Gastonia 138kV line?",
        expected_object_ids=["LINE_001"],
        expected_fragments=["0.3853", "138kV", "Charlotte-Gastonia"],
    ),
    EvalCase(
        question="What is the rated MVA of the transformer at Charlotte North substation?",
        expected_object_ids=["XFMR_SUB001_1"],
        expected_fragments=["600", "MVA"],
    ),
    EvalCase(
        question="How many MVAR does the capacitor bank at SUB_001 provide?",
        expected_object_ids=["CAP_SUB001_1"],
        expected_fragments=["50", "MVAR"],
    ),
    EvalCase(
        question="What voltage levels are present at the Charlotte North substation?",
        expected_object_ids=["VL_SUB001_138", "VL_SUB001_345"],
        expected_fragments=["138", "345"],
    ),
    EvalCase(
        question="What is the interrupting rating of the 138kV breaker on Line 001?",
        expected_object_ids=["BRK_LINE001_SUB001"],
        expected_fragments=["2000", "40"],
    ),
]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_hit_rate(results: List[dict], expected_ids: List[str]) -> float:
    retrieved_ids = {r["metadata"]["object_id"] for r in results}
    hits = sum(1 for eid in expected_ids if eid in retrieved_ids)
    return hits / len(expected_ids) if expected_ids else 0.0


def compute_reciprocal_rank(results: List[dict], expected_ids: List[str]) -> float:
    for rank, result in enumerate(results, 1):
        if result["metadata"]["object_id"] in expected_ids:
            return 1.0 / rank
    return 0.0


def compute_faithfulness(answer: str, expected_fragments: List[str]) -> float:
    """Simple substring faithfulness — fraction of expected fragments found in answer."""
    found = sum(1 for frag in expected_fragments if frag.lower() in answer.lower())
    return found / len(expected_fragments) if expected_fragments else 0.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(k: int = 5, verbose: bool = True) -> dict:
    collection = get_collection()
    chain = CIMRagChain(n_results=k)

    hit_rates, mrrs, faithfulness_scores = [], [], []

    for i, case in enumerate(EVAL_SET, 1):
        if verbose:
            print(f"\n[{i}/{len(EVAL_SET)}] {case.question}")

        hits = similarity_search(collection, case.question, n_results=k)
        hr = compute_hit_rate(hits, case.expected_object_ids)
        mrr = compute_reciprocal_rank(hits, case.expected_object_ids)

        result = chain.query(case.question)
        faith = compute_faithfulness(result["answer"], case.expected_fragments)

        hit_rates.append(hr)
        mrrs.append(mrr)
        faithfulness_scores.append(faith)

        if verbose:
            print(f"  Hit rate:     {hr:.2f}")
            print(f"  MRR:          {mrr:.2f}")
            print(f"  Faithfulness: {faith:.2f}")
            retrieved_names = [h["metadata"]["name"] for h in hits]
            print(f"  Retrieved:    {retrieved_names}")

    summary = {
        "k": k,
        "n_eval_cases": len(EVAL_SET),
        "hit_rate_at_k": round(sum(hit_rates) / len(hit_rates), 4),
        "mean_reciprocal_rank": round(sum(mrrs) / len(mrrs), 4),
        "mean_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
    }

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for key, val in summary.items():
        print(f"  {key}: {val}")

    # Save results for README badge / CI tracking
    with open("notebooks/eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to notebooks/eval_results.json")

    return summary


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    run_evaluation(k=5, verbose=True)
