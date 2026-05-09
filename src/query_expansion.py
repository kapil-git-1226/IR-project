import math
from collections import defaultdict
from typing import Dict, List, Tuple

from enhancements.ambiguity_detector import AmbiguityDetector
from enhancements.feedback_scorer import FeedbackQualityScorer
from enhancements.semantic_clusterer import SemanticFeedbackClusterer


class PseudoRelevanceFeedback:
    def __init__(self, index, preprocessor, doc_lookup: Dict[str, str] = None):
        self.index = index
        self.preprocessor = preprocessor
        self.doc_lookup = doc_lookup or {}

    def _extract_expansion_terms(self, feedback_doc_ids: List[str], top_m: int, exclude_terms: set = None) -> List[str]:
        """
        Extract the top-m expansion terms from the feedback documents.
        Scores each term by TF-IDF computed over the feedback document set.
        
        Args:
            feedback_doc_ids: List of document IDs to extract terms from.
            top_m: Number of top terms to return.
            exclude_terms: Set of terms to exclude (e.g., original query terms).
        
        Returns:
            List of top-m expansion terms (excluding any in exclude_terms).
        """
        feedback_set = set(feedback_doc_ids)
        exclude_terms = exclude_terms or set()
        term_scores = defaultdict(float)
        num_feedback = len(feedback_doc_ids)

        for term, postings in self.index.index.items():
            # Skip terms already in original query
            if term in exclude_terms:
                continue
            
            # Filter: Skip very common terms (appear in >50% of corpus)
            df = len(postings)
            if df > self.index.doc_count * 0.5:
                continue
            
            # Filter: Skip very rare terms (appear in <0.1% of corpus)
            if df < self.index.doc_count * 0.001:
                continue
                
            # Filter postings to only those in the feedback set
            feedback_postings = [(doc_id, tf) for doc_id, tf in postings if doc_id in feedback_set]
            if not feedback_postings:
                continue
            
            # Filter: Term must appear in at least 40% of feedback docs
            feedback_df = len(feedback_postings)
            if feedback_df < num_feedback * 0.4:
                continue

            # Aggregate TF across feedback documents
            total_tf = sum(tf for _, tf in feedback_postings)

            # IDF based on full corpus
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
        original_terms = self.preprocessor.preprocess(query)
        new_terms = self._extract_expansion_terms(
            feedback_doc_ids, 
            top_m=expansion_terms,
            exclude_terms=set(original_terms)  # Don't re-add original terms
        )

        # Step 4: Build expanded query — each term appears exactly once
        seen = set()
        all_terms = []
        for t in original_terms:
            if t not in seen:
                all_terms.append(t)
                seen.add(t)
        for t in new_terms:
            if t not in seen:
                all_terms.append(t)
                seen.add(t)
        expanded_query = ' '.join(all_terms)

        # Step 5: Final retrieval with expanded query
        final_results = retrieval_model.search(expanded_query, self.preprocessor, top_k=top_k)

        return final_results, expanded_query

    def _build_expanded_query(self, original_terms: List[str], new_terms: List[str]) -> str:
        seen = set()
        all_terms = []
        for term in original_terms + new_terms:
            if term not in seen:
                all_terms.append(term)
                seen.add(term)
        return " ".join(all_terms)

    def _resolve_feedback_docs(self, feedback_doc_ids: List[str]) -> List[str]:
        return [self.doc_lookup[doc_id] for doc_id in feedback_doc_ids if doc_id in self.doc_lookup]

    def search_with_prf_enhanced(
        self,
        query: str,
        retrieval_model,
        top_k: int = 10,
        use_ambiguity_detection: bool = True,
        use_quality_filtering: bool = True,
        use_semantic_clustering: bool = True,
    ) -> Tuple[List[Tuple[str, float]], str]:
        """
        Enhanced PRF pipeline:
        ambiguity detection -> feedback quality filtering -> semantic clustering.
        Falls back to baseline PRF if document texts are unavailable.
        """
        initial_results = retrieval_model.search(query, self.preprocessor, top_k=7)
        if not initial_results:
            return [], query

        feedback_doc_ids = [doc_id for doc_id, _ in initial_results]
        feedback_docs = self._resolve_feedback_docs(feedback_doc_ids)
        if not feedback_docs:
            return self.search_with_prf(query, retrieval_model, top_k=top_k)

        feedback_count = min(5, len(feedback_doc_ids))
        expansion_count = 5

        if use_ambiguity_detection:
            detector = AmbiguityDetector()
            ambiguity_result = detector.detect_ambiguity(feedback_docs)
            rec = ambiguity_result["recommended_params"]
            feedback_count = min(rec["feedback_docs"], len(feedback_doc_ids))
            expansion_count = rec["expansion_terms"]

        selected_ids = feedback_doc_ids[:feedback_count]
        selected_docs = self._resolve_feedback_docs(selected_ids)

        if use_quality_filtering and selected_docs:
            scorer = FeedbackQualityScorer()
            quality_result = scorer.score_feedback_documents(query, selected_docs)
            selected_docs = quality_result["filtered_docs"]

        if use_semantic_clustering and selected_docs:
            clusterer = SemanticFeedbackClusterer()
            cluster_result = clusterer.cluster_and_expand(
                selected_docs, query, num_expansion_terms=expansion_count
            )
            expansion_terms = [term for term, _ in cluster_result["expansion_terms"]]
        else:
            original_terms = self.preprocessor.preprocess(query)
            expansion_terms = self._extract_expansion_terms(
                selected_ids, top_m=expansion_count, exclude_terms=set(original_terms)
            )

        original_terms = self.preprocessor.preprocess(query)
        expanded_query = self._build_expanded_query(original_terms, expansion_terms)
        final_results = retrieval_model.search(expanded_query, self.preprocessor, top_k=top_k)
        return final_results, expanded_query
