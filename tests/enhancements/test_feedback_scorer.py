import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from enhancements.feedback_scorer import FeedbackQualityScorer, quick_filter_feedback


def test_clean_feedback():
    query = "oil prices middle east"
    feedback_docs = [
        "Oil prices in Middle East rose sharply following OPEC production cuts.",
        "Crude oil futures climbed as Middle Eastern tensions escalated.",
        "Saudi Arabia and UAE agreed to reduce oil output affecting prices.",
        "Brent crude from Middle East reached $85 per barrel.",
        "Middle East oil producers navigate geopolitical risks and pricing.",
    ]
    scorer = FeedbackQualityScorer(quality_threshold=0.5)
    result = scorer.score_feedback_documents(query, feedback_docs)
    assert result["num_kept"] >= 2
    assert result["avg_quality_kept"] >= 0.0


def test_noisy_feedback():
    query = "stock market crash 1987"
    feedback_docs = [
        "The 1987 stock market crash known as Black Monday saw Dow Jones drop 22 percent.",
        "Stock market volatility increased during October 1987 financial crisis.",
        "Weather patterns in 1987 included unusual hurricane activity.",
        "Recipe for chocolate cake with three layers and frosting.",
        "Market crash of 1987 triggered circuit breakers and trading halts.",
        "Program trading and portfolio insurance blamed for 1987 crash severity.",
    ]
    scorer = FeedbackQualityScorer(quality_threshold=0.5)
    result = scorer.score_feedback_documents(query, feedback_docs)
    assert result["num_filtered"] >= 1


def test_ambiguous_query_filtering():
    query = "python"
    feedback_docs = [
        "Python programming language is widely used in machine learning and data science.",
        "Python 3.11 introduced significant performance improvements and new syntax.",
        "Python snakes are non-venomous constrictors found in tropical regions.",
        "Python frameworks like Django and Flask enable rapid web development.",
        "Learning Python is essential for modern software engineering careers.",
        "The reticulated python is one of the longest snake species.",
    ]
    scorer = FeedbackQualityScorer(quality_threshold=0.45)
    result = scorer.score_feedback_documents(query, feedback_docs)
    assert result["num_filtered"] > 0


def test_minimum_docs_safety():
    query = "machine learning"
    feedback_docs = [
        "Machine learning algorithms require large datasets for training.",
        "Weather forecast models use statistical analysis.",
        "Recipe recommendations based on user preferences.",
        "Image recognition using neural networks.",
    ]
    scorer = FeedbackQualityScorer(quality_threshold=0.9, min_feedback_docs=2)
    result = scorer.score_feedback_documents(query, feedback_docs)
    assert result["num_kept"] >= scorer.min_docs


def test_quick_filter_utility():
    query = "economic recession"
    feedback_docs = [
        "Economic recession indicators include rising unemployment.",
        "GDP contraction signals potential recession.",
        "Basketball game highlights from last night.",
        "Central banks lower interest rates during recession.",
    ]
    filtered = quick_filter_feedback(query, feedback_docs, threshold=0.5)
    assert len(filtered) < len(feedback_docs)
    assert len(filtered) >= 2

