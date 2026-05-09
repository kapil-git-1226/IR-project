import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from enhancements.semantic_clusterer import SemanticFeedbackClusterer, quick_cluster_expansion


def test_homogeneous_feedback():
    feedback_docs = [
        "Oil prices in Middle East rose sharply following OPEC production cuts.",
        "Crude oil futures climbed as Middle Eastern tensions escalated.",
        "Saudi Arabia and UAE agreed to reduce oil output affecting prices.",
        "Brent crude from Middle East reached $85 per barrel.",
        "Middle East oil producers navigate geopolitical risks and pricing.",
    ]
    clusterer = SemanticFeedbackClusterer()
    result = clusterer.cluster_feedback(feedback_docs)
    assert result["num_clusters"] >= 1
    assert result["coverage"] > 0


def test_heterogeneous_feedback():
    feedback_docs = [
        "The Jaguar XF is a luxury sedan with advanced safety features.",
        "Jaguar Land Rover announced new electric vehicle lineup.",
        "Jaguar's latest SUV model features all-wheel drive.",
        "Jaguars are large cats native to Americas.",
        "The jaguar is an apex predator that hunts fish.",
        "Jaguar populations in rainforests face extinction threats.",
    ]
    clusterer = SemanticFeedbackClusterer(similarity_threshold=0.4)
    result = clusterer.cluster_feedback(feedback_docs)
    assert result["num_clusters"] >= 1


def test_term_extraction_from_cluster():
    feedback_docs = [
        "Python programming language is widely used in data science.",
        "Python 3.11 introduced significant performance improvements.",
        "Python frameworks like Django and Flask enable web development.",
        "Python snakes are non-venomous constrictors.",
        "The python is a snake species.",
    ]
    clusterer = SemanticFeedbackClusterer(similarity_threshold=0.4)
    result = clusterer.cluster_and_expand(feedback_docs, "python", num_expansion_terms=5)
    words = [term for term, _ in result["expansion_terms"]]
    assert isinstance(words, list)
    assert "python" not in words if words else True


def test_edge_case_too_few_docs():
    clusterer = SemanticFeedbackClusterer()
    result = clusterer.cluster_feedback(["Single document about stock market."])
    assert result["num_clusters"] == 1
    assert "note" in result


def test_quick_expansion_utility():
    feedback_docs = [
        "Apple Inc released iPhone 15 with new features.",
        "Apple announced M3 chip for MacBooks.",
        "Apple stock price rose after earnings report.",
        "Apple pie recipe with cinnamon and sugar.",
    ]
    expansion_terms = quick_cluster_expansion(feedback_docs, "Apple", num_terms=5)
    assert isinstance(expansion_terms, list)
    assert all(isinstance(t, str) for t in expansion_terms)

