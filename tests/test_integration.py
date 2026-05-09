import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from enhancements.ambiguity_detector import AmbiguityDetector
from enhancements.feedback_scorer import FeedbackQualityScorer
from enhancements.semantic_clusterer import SemanticFeedbackClusterer


def test_full_pipeline():
    query = "stock market crash 1987"
    feedback_docs = [
        "The 1987 stock market crash known as Black Monday saw Dow Jones drop 22 percent.",
        "Stock market volatility increased during October 1987 financial crisis.",
        "Weather patterns in 1987 included unusual hurricane activity.",
        "Market crash of 1987 triggered circuit breakers and trading halts.",
        "Program trading and portfolio insurance blamed for 1987 crash severity.",
    ]

    detector = AmbiguityDetector()
    amb = detector.detect_ambiguity(feedback_docs)
    assert "classification" in amb

    scorer = FeedbackQualityScorer()
    quality = scorer.score_feedback_documents(query, feedback_docs)
    assert quality["num_kept"] >= 2

    clusterer = SemanticFeedbackClusterer()
    cluster = clusterer.cluster_and_expand(quality["filtered_docs"], query, num_expansion_terms=5)
    assert "cluster_result" in cluster
    assert isinstance(cluster["expansion_terms"], list)

