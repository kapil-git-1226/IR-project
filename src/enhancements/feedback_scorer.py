"""
Feedback quality scoring and filtering for PRF.
"""

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeedbackQualityScorer:
    """Score feedback docs and remove noisy items."""

    def __init__(
        self,
        quality_threshold: float = 0.5,
        relevance_weight: float = 0.6,
        coherence_weight: float = 0.4,
        min_feedback_docs: int = 2,
    ):
        assert abs(relevance_weight + coherence_weight - 1.0) < 0.01
        self.threshold = quality_threshold
        self.alpha = relevance_weight
        self.beta = coherence_weight
        self.min_docs = min_feedback_docs

    def score_feedback_documents(
        self, query: str, feedback_docs: List[str], return_details: bool = False
    ) -> Dict:
        if len(feedback_docs) < self.min_docs:
            return {
                "filtered_docs": feedback_docs,
                "num_original": len(feedback_docs),
                "num_filtered": 0,
                "num_kept": len(feedback_docs),
                "avg_quality_kept": 1.0,
                "note": "Insufficient feedback docs for quality filtering",
            }

        try:
            vec = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
            vectors = vec.fit_transform([query] + feedback_docs)
            query_v = vectors[0:1]
            doc_v = vectors[1:]
        except Exception as exc:
            return {
                "filtered_docs": feedback_docs,
                "num_original": len(feedback_docs),
                "num_filtered": 0,
                "num_kept": len(feedback_docs),
                "error": str(exc),
            }

        relevance = cosine_similarity(query_v, doc_v)[0]
        coherence = []
        for i in range(len(feedback_docs)):
            this_doc = doc_v[i : i + 1]
            others = np.vstack(
                [
                    doc_v[:i].toarray() if i > 0 else np.empty((0, doc_v.shape[1])),
                    doc_v[i + 1 :].toarray()
                    if i < len(feedback_docs) - 1
                    else np.empty((0, doc_v.shape[1])),
                ]
            )
            if others.shape[0] == 0:
                coherence.append(1.0)
            else:
                coherence.append(float(np.mean(cosine_similarity(this_doc, others)[0])))

        coherence_arr = np.array(coherence)
        quality = self.alpha * relevance + self.beta * coherence_arr
        keep_mask = quality >= self.threshold

        if int(np.sum(keep_mask)) < self.min_docs:
            top_indices = np.argsort(quality)[-self.min_docs :]
            keep_mask = np.zeros(len(feedback_docs), dtype=bool)
            keep_mask[top_indices] = True

        filtered_docs = [doc for doc, keep in zip(feedback_docs, keep_mask) if keep]
        kept_scores = quality[keep_mask]
        dropped_scores = quality[~keep_mask]

        result = {
            "filtered_docs": filtered_docs,
            "num_original": len(feedback_docs),
            "num_filtered": len(feedback_docs) - len(filtered_docs),
            "num_kept": len(filtered_docs),
            "avg_quality_kept": float(np.mean(kept_scores)) if len(kept_scores) else 0.0,
            "avg_quality_filtered": float(np.mean(dropped_scores)) if len(dropped_scores) else 0.0,
        }

        if return_details:
            details = []
            for i, (doc, rel, coh, qual, keep) in enumerate(
                zip(feedback_docs, relevance, coherence_arr, quality, keep_mask)
            ):
                details.append(
                    {
                        "doc_index": i,
                        "relevance": float(rel),
                        "coherence": float(coh),
                        "quality": float(qual),
                        "kept": bool(keep),
                        "doc_preview": doc[:100] + "..." if len(doc) > 100 else doc,
                    }
                )
            result["scores"] = details

        return result

    def explain_filtering(self, scoring_result: Dict) -> str:
        interpretation = (
            f"Filtered out {scoring_result['num_filtered']} low-quality documents."
            if scoring_result["num_filtered"] > 0
            else "All feedback documents passed quality threshold."
        )
        return (
            "Feedback Quality Filtering Analysis:\n"
            "------------------------------------\n"
            f"Original Feedback Documents: {scoring_result['num_original']}\n"
            f"Documents Kept: {scoring_result['num_kept']}\n"
            f"Documents Filtered Out: {scoring_result['num_filtered']}\n\n"
            "Quality Scores:\n"
            f"  - Kept Documents: {scoring_result['avg_quality_kept']:.3f} (avg)\n"
            f"  - Filtered Documents: {scoring_result.get('avg_quality_filtered', 0.0):.3f} (avg)\n"
            f"  - Threshold: {self.threshold}\n\n"
            "Scoring Formula:\n"
            f"  quality = {self.alpha} * relevance + {self.beta} * coherence\n\n"
            f"Interpretation:\n{interpretation}"
        )


def quick_filter_feedback(query: str, feedback_docs: List[str], threshold: float = 0.5) -> List[str]:
    scorer = FeedbackQualityScorer(quality_threshold=threshold)
    result = scorer.score_feedback_documents(query, feedback_docs)
    return result["filtered_docs"]

