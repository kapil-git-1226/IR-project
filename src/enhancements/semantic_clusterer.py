"""
Semantic clustering of feedback docs for PRF expansion.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticFeedbackClusterer:
    """Cluster feedback docs and extract terms from dominant cluster."""

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        min_cluster_size: int = 2,
        max_expansion_terms: int = 10,
    ):
        self.threshold = similarity_threshold
        self.min_size = min_cluster_size
        self.max_terms = max_expansion_terms

    def cluster_feedback(self, feedback_docs: List[str], return_details: bool = False) -> Dict:
        if len(feedback_docs) < self.min_size:
            return {
                "largest_cluster": feedback_docs,
                "cluster_indices": list(range(len(feedback_docs))),
                "num_clusters": 1,
                "cluster_sizes": [len(feedback_docs)],
                "docs_in_largest": len(feedback_docs),
                "coverage": 1.0,
                "note": "Insufficient documents for clustering",
            }

        try:
            vec = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2), min_df=1
            )
            doc_vectors = vec.fit_transform(feedback_docs)
        except Exception as exc:
            return {
                "largest_cluster": feedback_docs,
                "cluster_indices": list(range(len(feedback_docs))),
                "num_clusters": 1,
                "cluster_sizes": [len(feedback_docs)],
                "docs_in_largest": len(feedback_docs),
                "coverage": 1.0,
                "error": str(exc),
            }

        sim = cosine_similarity(doc_vectors)
        graph = defaultdict(set)
        for i in range(len(feedback_docs)):
            for j in range(i + 1, len(feedback_docs)):
                if sim[i, j] >= self.threshold:
                    graph[i].add(j)
                    graph[j].add(i)

        clusters: List[List[int]] = []
        visited = set()
        for start in range(len(feedback_docs)):
            if start in visited:
                continue
            queue = [start]
            cluster = []
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            clusters.append(cluster)

        sizes = [len(c) for c in clusters]
        largest = clusters[int(np.argmax(sizes))] if clusters else list(range(len(feedback_docs)))
        largest_docs = [feedback_docs[i] for i in largest]

        result = {
            "largest_cluster": largest_docs,
            "cluster_indices": largest,
            "num_clusters": len(clusters),
            "cluster_sizes": sizes,
            "docs_in_largest": len(largest_docs),
            "coverage": len(largest_docs) / len(feedback_docs),
        }
        if return_details:
            result["similarity_matrix"] = sim
            result["all_clusters"] = clusters
            result["graph"] = dict(graph)
        return result

    def extract_expansion_terms(
        self, cluster_docs: List[str], original_query_terms: Set[str], num_terms: int = None
    ) -> List[Tuple[str, float]]:
        if num_terms is None:
            num_terms = self.max_terms
        if not cluster_docs:
            return []
        try:
            vec = TfidfVectorizer(
                max_features=500,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8,
            )
            matrix = vec.fit_transform(cluster_docs)
            names = vec.get_feature_names_out()
            scores = np.asarray(matrix.sum(axis=0)).flatten()
            pairs = [
                (term, score)
                for term, score in zip(names, scores)
                if term not in original_query_terms
            ]
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:num_terms]
        except Exception:
            return []

    def cluster_and_expand(
        self, feedback_docs: List[str], original_query: str, num_expansion_terms: int = None
    ) -> Dict:
        cluster_result = self.cluster_feedback(feedback_docs)
        original_terms = set(original_query.lower().split())
        expansion = self.extract_expansion_terms(
            cluster_result["largest_cluster"], original_terms, num_terms=num_expansion_terms
        )
        return {
            "cluster_result": cluster_result,
            "expansion_terms": expansion,
            "original_query_terms": original_terms,
            "num_new_terms": len(expansion),
        }

    def explain_clustering(self, cluster_result: Dict) -> str:
        total_docs = (
            cluster_result["docs_in_largest"] / cluster_result["coverage"]
            if cluster_result["coverage"] > 0
            else 0
        )
        return (
            "Semantic Feedback Clustering Analysis:\n"
            "-------------------------------------\n"
            f"Total Feedback Documents: {total_docs:.0f}\n"
            f"Clusters Identified: {cluster_result['num_clusters']}\n"
            f"Cluster Sizes: {cluster_result['cluster_sizes']}\n"
        )


def quick_cluster_expansion(feedback_docs: List[str], query: str, num_terms: int = 10) -> List[str]:
    clusterer = SemanticFeedbackClusterer()
    result = clusterer.cluster_and_expand(feedback_docs, query, num_terms)
    return [term for term, _ in result["expansion_terms"]]

