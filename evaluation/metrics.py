from typing import List, Set


def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int = 10) -> float:
    """
    Calculate Precision at K.

    Args:
        retrieved_docs: Ranked list of retrieved document IDs.
        relevant_docs:  Set of ground-truth relevant document IDs.
        k:              Number of top results to consider.

    Returns:
        Proportion of the top-k retrieved documents that are relevant. (0.0 – 1.0)
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_docs[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return hits / k


def recall(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
    """
    Calculate Recall.

    Args:
        retrieved_docs: Ranked list of retrieved document IDs.
        relevant_docs:  Set of ground-truth relevant document IDs.

    Returns:
        Proportion of all relevant documents that were retrieved. (0.0 – 1.0)
        Returns 0.0 if relevant_docs is empty.
    """
    if not relevant_docs:
        return 0.0
    retrieved_set = set(retrieved_docs)
    hits = len(retrieved_set & relevant_docs)
    return hits / len(relevant_docs)


def average_precision(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
    """
    Calculate Average Precision (AP) for a single query.

    Precision is computed at every rank position where a relevant document
    is found, then averaged over all relevant documents.

    Args:
        retrieved_docs: Ranked list of retrieved document IDs.
        relevant_docs:  Set of ground-truth relevant document IDs.

    Returns:
        Average Precision score. (0.0 – 1.0)
        Returns 0.0 if relevant_docs is empty.
    """
    if not relevant_docs:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            hits += 1
            precision_sum += hits / rank  # Precision at this rank

    return precision_sum / len(relevant_docs)


def mean_average_precision(ap_scores: List[float]) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.

    Args:
        ap_scores: List of Average Precision scores, one per query.

    Returns:
        Mean of all AP scores. (0.0 – 1.0)
        Returns 0.0 if ap_scores is empty.
    """
    if not ap_scores:
        return 0.0
    return sum(ap_scores) / len(ap_scores)
