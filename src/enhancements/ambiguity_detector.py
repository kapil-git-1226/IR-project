"""
Adaptive ambiguity detection for PRF.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AmbiguityDetector:
    """Classify feedback coherence as ambiguous vs specific."""

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        conservative_feedback_docs: int = 3,
        conservative_expansion_terms: int = 3,
        aggressive_feedback_docs: int = 7,
        aggressive_expansion_terms: int = 8,
    ):
        self.threshold = similarity_threshold
        self.params = {
            "ambiguous": {
                "feedback_docs": conservative_feedback_docs,
                "expansion_terms": conservative_expansion_terms,
                "alpha": 1.5,
                "beta": 0.3,
            },
            "specific": {
                "feedback_docs": aggressive_feedback_docs,
                "expansion_terms": aggressive_expansion_terms,
                "alpha": 1.0,
                "beta": 0.7,
            },
        }

    def detect_ambiguity(self, feedback_docs: List[str], return_details: bool = False) -> Dict:
        if len(feedback_docs) < 2:
            return {
                "is_ambiguous": False,
                "coherence_score": 1.0,
                "classification": "SPECIFIC",
                "recommended_params": self.params["specific"],
                "note": "Insufficient feedback documents for ambiguity detection",
            }

        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
            doc_vectors = vectorizer.fit_transform(feedback_docs)
        except Exception as exc:
            return {
                "is_ambiguous": True,
                "coherence_score": 0.0,
                "classification": "AMBIGUOUS",
                "recommended_params": self.params["ambiguous"],
                "error": str(exc),
            }

        sim = cosine_similarity(doc_vectors)
        tri = np.triu_indices_from(sim, k=1)
        pairwise = sim[tri]
        coherence = float(np.mean(pairwise))

        is_ambiguous = coherence < self.threshold
        label = "AMBIGUOUS" if is_ambiguous else "SPECIFIC"
        params = self.params["ambiguous" if is_ambiguous else "specific"]

        result = {
            "is_ambiguous": is_ambiguous,
            "coherence_score": coherence,
            "classification": label,
            "recommended_params": params,
            "num_feedback_docs": len(feedback_docs),
        }
        if return_details:
            result["similarity_matrix"] = sim
            result["pairwise_similarities"] = pairwise
        return result

    def explain_decision(self, detection_result: Dict) -> str:
        score = detection_result["coherence_score"]
        classification = detection_result["classification"]
        params = detection_result["recommended_params"]
        interpretation = (
            f"The feedback documents show LOW semantic coherence (< {self.threshold}), "
            "indicating multiple interpretations. Using conservative expansion."
            if detection_result["is_ambiguous"]
            else f"The feedback documents show HIGH semantic coherence (>= {self.threshold}), "
            "indicating a focused topic. Using aggressive expansion."
        )
        return (
            "Ambiguity Detection Analysis:\n"
            "-----------------------------\n"
            f"Coherence Score: {score:.3f} (threshold: {self.threshold})\n"
            f"Classification: {classification}\n\n"
            f"Interpretation:\n{interpretation}\n\n"
            "Recommended Parameters:\n"
            f"  - Feedback Documents: {params['feedback_docs']}\n"
            f"  - Expansion Terms: {params['expansion_terms']}\n"
            f"  - Original Query Weight (alpha): {params['alpha']}\n"
            f"  - Expansion Weight (beta): {params['beta']}"
        )


def quick_ambiguity_check(feedback_texts: List[str]) -> Tuple[bool, float]:
    detector = AmbiguityDetector()
    result = detector.detect_ambiguity(feedback_texts)
    return result["is_ambiguous"], result["coherence_score"]

