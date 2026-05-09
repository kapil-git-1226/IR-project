import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from enhancements.ambiguity_detector import AmbiguityDetector, quick_ambiguity_check


def test_ambiguous_query():
    feedback_docs = [
        "The Jaguar XF is a luxury sedan with advanced safety features.",
        "Jaguars are large cats native to Americas with powerful bite force.",
        "Jaguar Land Rover announced new electric vehicle lineup.",
        "The jaguar is an apex predator that hunts fish and caimans.",
        "Jaguar's latest SUV model features all-wheel drive and premium interior.",
    ]
    detector = AmbiguityDetector(similarity_threshold=0.04)
    result = detector.detect_ambiguity(feedback_docs)
    assert result["is_ambiguous"] is True
    assert result["coherence_score"] < 0.4


def test_specific_query():
    feedback_docs = [
        "Oil prices in the Middle East rose sharply following OPEC production cuts.",
        "Crude oil futures climbed as Middle Eastern tensions escalated.",
        "Saudi Arabia and UAE agreed to reduce oil output affecting global prices.",
        "Brent crude from Middle East reached $85 per barrel amid supply concerns.",
        "Middle East oil producers navigate geopolitical risks and price volatility.",
    ]
    detector = AmbiguityDetector(similarity_threshold=0.04)
    result = detector.detect_ambiguity(feedback_docs)
    assert result["is_ambiguous"] is False
    assert result["coherence_score"] >= 0.04


def test_edge_case_single_doc():
    detector = AmbiguityDetector()
    result = detector.detect_ambiguity(["Single document about stock market crash."])
    assert result["is_ambiguous"] is False
    assert "note" in result


def test_explanation_generation():
    detector = AmbiguityDetector()
    result = detector.detect_ambiguity(
        [
            "Turkey is a country in both Europe and Asia.",
            "Roasted turkey is a traditional Thanksgiving dish.",
            "Turkish economy faces inflation challenges.",
        ]
    )
    explanation = detector.explain_decision(result)
    assert len(explanation) > 100
    assert result["classification"] in explanation


def test_quick_check_function():
    is_ambiguous, coherence_score = quick_ambiguity_check(
        [
            "Python programming language is widely used in data science.",
            "Python snakes are non-venomous constrictors.",
            "Python 3.11 introduced performance improvements.",
        ]
    )
    assert isinstance(is_ambiguous, bool)
    assert isinstance(coherence_score, float)

