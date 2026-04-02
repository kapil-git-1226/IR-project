import math
from collections import defaultdict
from typing import List, Tuple


class PseudoRelevanceFeedback:
    def __init__(self, index, preprocessor):
        self.index = index
        self.preprocessor = preprocessor

    def _extract_expansion_terms(self, feedback_doc_ids: List[str], top_m: int) -> List[str]:
        """
        Extract the top-m expansion terms from the feedback documents.
        Scores each term by TF-IDF computed over the feedback document set.
        """
        feedback_set = set(feedback_doc_ids)
        term_scores = defaultdict(float)

        for term, postings in self.index.index.items():
            # Filter postings to only those in the feedback set
            feedback_postings = [(doc_id, tf) for doc_id, tf in postings if doc_id in feedback_set]
            if not feedback_postings:
                continue

            # Aggregate TF across feedback documents
            total_tf = sum(tf for _, tf in feedback_postings)

            # IDF based on full corpus
            df = len(postings)
            idf = math.log(self.index.doc_count / df)

            term_scores[term] = total_tf * idf

        # Sort by score descending and return top_m term strings
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:top_m]]

    def search_with_prf(
        self,
        query: str,
        retrieval_model,
        top_k: int = 10,
        feedback_docs: int = 5,
        expansion_terms: int = 5
    ) -> Tuple[List[Tuple[str, float]], str]:
        """
        Perform retrieval with Pseudo-Relevance Feedback.

        Steps:
            1. Run initial retrieval with the original query.
            2. Take the top `feedback_docs` as pseudo-relevant documents.
            3. Extract the top `expansion_terms` from those documents.
            4. Expand the original query with those terms.
            5. Run a second retrieval with the expanded query.

        Returns:
            Tuple of (final_results, expanded_query_string)
        """
        # Step 1: Initial retrieval
        initial_results = retrieval_model.search(query, self.preprocessor, top_k=feedback_docs)
        if not initial_results:
            return [], query

        # Step 2: Collect feedback document IDs
        feedback_doc_ids = [doc_id for doc_id, _ in initial_results]

        # Step 3: Extract top expansion terms from feedback documents
        new_terms = self._extract_expansion_terms(feedback_doc_ids, top_m=expansion_terms)

        # Step 4: Build expanded query string
        original_terms = self.preprocessor.preprocess(query)
        all_terms = original_terms + [t for t in new_terms if t not in original_terms]
        expanded_query = ' '.join(all_terms)

        # Step 5: Final retrieval with expanded query
        final_results = retrieval_model.search(expanded_query, self.preprocessor, top_k=top_k)

        return final_results, expanded_query
