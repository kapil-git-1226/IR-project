import math
from typing import List, Tuple


class TFIDFRetrieval:
    def __init__(self, index):
        self.index = index

    def search(self, query: str, preprocessor, top_k: int = 10) -> List[Tuple[str, float]]:
        query_terms = preprocessor.preprocess(query)
        if not query_terms:
            return []

        # Build postings dict and collect candidate docs
        postings_dict = {}
        candidate_docs = set()

        for term in query_terms:
            postings = self.index.get_postings(term)
            if postings:
                postings_dict[term] = {doc_id: tf for doc_id, tf in postings}
                candidate_docs.update(postings_dict[term].keys())

        if not candidate_docs:
            return []

        N = self.index.doc_count
        scores = {}

        for doc_id in candidate_docs:
            score = 0.0
            for term in query_terms:
                if term not in postings_dict:
                    continue
                tf = postings_dict[term].get(doc_id, 0)
                if tf == 0:
                    continue
                df = self.index.get_document_frequency(term)
                if df == 0:
                    continue
                idf = math.log(N / df)
                score += tf * idf
            scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class BM25Retrieval:
    def __init__(self, index, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def search(self, query: str, preprocessor, top_k: int = 10) -> List[Tuple[str, float]]:
        query_terms = preprocessor.preprocess(query)
        if not query_terms:
            return []

        # Build postings dict and collect candidate docs
        postings_dict = {}
        candidate_docs = set()

        for term in query_terms:
            postings = self.index.get_postings(term)
            if postings:
                postings_dict[term] = {doc_id: tf for doc_id, tf in postings}
                candidate_docs.update(postings_dict[term].keys())

        if not candidate_docs:
            return []

        N = self.index.doc_count
        avgdl = self.index.avg_doc_length
        scores = {}

        for doc_id in candidate_docs:
            score = 0.0
            doc_length = self.index.doc_lengths.get(doc_id, 0)

            for term in query_terms:
                if term not in postings_dict:
                    continue
                tf = postings_dict[term].get(doc_id, 0)
                if tf == 0:
                    continue
                df = self.index.get_document_frequency(term)
                if df == 0:
                    continue

                # BM25 IDF (Robertson variant)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

                # BM25 TF normalization
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avgdl))

                score += idf * (numerator / denominator)

            scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
