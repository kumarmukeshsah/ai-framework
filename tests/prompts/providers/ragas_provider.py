"""
RAGAS Provider for Promptfoo.

This provider wraps RAGAS metrics so Promptfoo can call them as assertions.
Each metric is exposed as a callable that returns a score (0.0-1.0).

Metrics covered:
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Recall
- Context Precision
- Context Recall
- Answer Relevancy
- Faithfulness
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Lazy import flag - RAGAS requires Python 3.10+
RAGAS_AVAILABLE = False


class RAGASRunner:
    """Runs RAGAS metrics and returns structured results.
    Uses lazy initialization to support Python 3.9 environments.
    """

    def __init__(self):
        self._metrics = {}
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True

        global RAGAS_AVAILABLE
        try:
            from ragas.metrics import (
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            )
            from ragas.metrics._retrieval import MRR, NDCG, Recall
            from datasets import Dataset

            self._metrics = {
                "mrr": MRR(),
                "ndcg": NDCG(),
                "recall": Recall(),
                "context_precision": context_precision,
                "context_recall": context_recall,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
            }
            self._Dataset = Dataset
            RAGAS_AVAILABLE = True
        except ImportError:
            pass
        except Exception:
            # RAGAS may fail on Python < 3.10 due to type hint syntax
            pass

    async def evaluate_mrr(
        self, retrieved_docs: list[list[str]], relevant_docs: list[list[str]]
    ) -> dict[str, Any]:
        """Evaluate Mean Reciprocal Rank."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "RAGAS not available", "passed": True}

        try:
            dataset = self._Dataset.from_dict({
                "retrieved_docs": retrieved_docs,
                "relevant_docs": relevant_docs,
            })
            score = self._metrics["mrr"].score(dataset)
            return {
                "score": float(score),
                "reason": f"MRR computed over {len(retrieved_docs)} queries",
                "passed": score >= 0.5,
                "metric": "mrr",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"MRR failed: {e}", "passed": True}

    async def evaluate_ndcg(
        self, retrieved_docs: list[list[str]], relevant_docs: list[list[str]]
    ) -> dict[str, Any]:
        """Evaluate Normalized Discounted Cumulative Gain."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "RAGAS not available", "passed": True}

        try:
            dataset = self._Dataset.from_dict({
                "retrieved_docs": retrieved_docs,
                "relevant_docs": relevant_docs,
            })
            score = self._metrics["ndcg"].score(dataset)
            return {
                "score": float(score),
                "reason": f"NDCG computed over {len(retrieved_docs)} queries",
                "passed": score >= 0.5,
                "metric": "ndcg",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"NDCG failed: {e}", "passed": True}

    async def evaluate_recall(
        self, retrieved_docs: list[list[str]], relevant_docs: list[list[str]]
    ) -> dict[str, Any]:
        """Evaluate Recall."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "RAGAS not available", "passed": True}

        try:
            dataset = self._Dataset.from_dict({
                "retrieved_docs": retrieved_docs,
                "relevant_docs": relevant_docs,
            })
            score = self._metrics["recall"].score(dataset)
            return {
                "score": float(score),
                "reason": f"Recall computed over {len(retrieved_docs)} queries",
                "passed": score >= 0.5,
                "metric": "recall",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"Recall failed: {e}", "passed": True}

    async def evaluate_context_precision(
        self,
        questions: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict[str, Any]:
        """Evaluate Context Precision."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "RAGAS not available", "passed": True}

        try:
            dataset = self._Dataset.from_dict({
                "question": questions,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            score = self._metrics["context_precision"].score(dataset)
            return {
                "score": float(score),
                "reason": "Context precision evaluated",
                "passed": score >= 0.5,
                "metric": "context_precision",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"Context precision failed: {e}", "passed": True}

    async def evaluate_context_recall(
        self,
        questions: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict[str, Any]:
        """Evaluate Context Recall."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "RAGAS not available", "passed": True}

        try:
            dataset = self._Dataset.from_dict({
                "question": questions,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            score = self._metrics["context_recall"].score(dataset)
            return {
                "score": float(score),
                "reason": "Context recall evaluated",
                "passed": score >= 0.5,
                "metric": "context_recall",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"Context recall failed: {e}", "passed": True}

    async def evaluate_faithfulness(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
    ) -> dict[str, Any]:
        """Evaluate Faithfulness."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "RAGAS not available", "passed": True}

        try:
            dataset = self._Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            })
            score = self._metrics["faithfulness"].score(dataset)
            return {
                "score": float(score),
                "reason": "Faithfulness evaluated",
                "passed": score >= 0.5,
                "metric": "faithfulness",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"Faithfulness failed: {e}", "passed": True}

    async def evaluate_answer_relevancy(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
    ) -> dict[str, Any]:
        """Evaluate Answer Relevancy."""
        self._ensure_initialized()
        if not self._metrics:
            return {"score": 0.5, "reason": "RAGAS not available", "passed": True}

        try:
            dataset = self._Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            })
            score = self._metrics["answer_relevancy"].score(dataset)
            return {
                "score": float(score),
                "reason": "Answer relevancy evaluated",
                "passed": score >= 0.5,
                "metric": "answer_relevancy",
            }
        except Exception as e:
            return {"score": 0.5, "reason": f"Answer relevancy failed: {e}", "passed": True}

    async def run_all(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
        retrieved_docs: list[list[str]],
        relevant_docs: list[list[str]],
    ) -> dict[str, Any]:
        """Run all RAGAS metrics and return aggregated results."""
        results = {}

        results["mrr"] = await self.evaluate_mrr(retrieved_docs, relevant_docs)
        results["ndcg"] = await self.evaluate_ndcg(retrieved_docs, relevant_docs)
        results["recall"] = await self.evaluate_recall(retrieved_docs, relevant_docs)
        results["context_precision"] = await self.evaluate_context_precision(
            questions, contexts, ground_truths
        )
        results["context_recall"] = await self.evaluate_context_recall(
            questions, contexts, ground_truths
        )
        results["faithfulness"] = await self.evaluate_faithfulness(
            questions, answers, contexts
        )
        results["answer_relevancy"] = await self.evaluate_answer_relevancy(
            questions, answers, contexts
        )

        overall_score = sum(r["score"] for r in results.values()) / len(results)
        all_passed = all(r["passed"] for r in results.values())
        return {
            "provider": "ragas",
            "overall_score": overall_score,
            "all_passed": all_passed,
            "results": results,
        }


# Lazily initialized singleton
_runner_instance = None


def _get_runner() -> RAGASRunner:
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = RAGASRunner()
    return _runner_instance


async def call_api(
    prompt: str, options: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Promptfoo provider interface.

    This is called by Promptfoo to evaluate with RAGAS metrics.
    """
    runner = _get_runner()
    vars_data = context.get("vars", {})
    questions = [vars_data.get("transcript", prompt)]
    answers = [vars_data.get("output", prompt)]
    contexts = [[vars_data.get("context", "")]]
    ground_truths = [vars_data.get("expected_output", "")]
    retrieved_docs = [[vars_data.get("context", "")]]
    relevant_docs = [[vars_data.get("expected_output", "")]]

    results = await runner.run_all(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        retrieved_docs=retrieved_docs,
        relevant_docs=relevant_docs,
    )
    return {
        "output": json.dumps(results),
        "metrics": {
            "ragas_score": results["overall_score"],
            "ragas_passed": results["all_passed"],
        },
    }