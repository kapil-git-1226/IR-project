import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


class InvertedIndex:
    def __init__(self):
        self.index: defaultdict = defaultdict(list)  # term -> [(doc_id, tf), ...]
        self.doc_lengths: Dict[str, int] = {}        # doc_id -> token count
        self.doc_count: int = 0
        self.avg_doc_length: float = 0.0

    def build_index(self, documents: List[Dict], preprocessor) -> None:
        """
        Build the inverted index from a list of documents.

        Args:
            documents: List of dicts with keys 'doc_id' and 'text'.
            preprocessor: TextPreprocessor instance with a preprocess() method.
        """
        total_tokens = 0

        for doc in documents:
            doc_id = doc['doc_id']
            tokens = preprocessor.preprocess(doc['text'])

            if not tokens:
                continue

            # Term frequencies for this document
            term_freq = Counter(tokens)

            # Store document length
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            total_tokens += doc_length

            # Add to inverted index
            for term, tf in term_freq.items():
                self.index[term].append((doc_id, tf))

        self.doc_count = len(self.doc_lengths)
        self.avg_doc_length = total_tokens / self.doc_count if self.doc_count > 0 else 0.0

        print(f"[INDEX] Built index: {len(self.index)} unique terms, {self.doc_count} documents")
        print(f"[INDEX] Average document length: {self.avg_doc_length:.2f} tokens")

    def get_postings(self, term: str) -> List[Tuple[str, int]]:
        """
        Return postings list for a term.

        Args:
            term: A preprocessed (stemmed) term.

        Returns:
            List of (doc_id, tf) tuples.
        """
        return self.index.get(term, [])

    def get_document_frequency(self, term: str) -> int:
        """
        Return the number of documents containing the given term.

        Args:
            term: A preprocessed (stemmed) term.

        Returns:
            Document frequency (int).
        """
        return len(self.index.get(term, []))

    def save(self, file_path: str) -> None:
        """
        Serialize the index to disk using pickle.

        Args:
            file_path: Full path to the output .pkl file.
        """
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

        print(f"[SAVED] Index saved to {file_path}")

    @staticmethod
    def load(file_path: str) -> 'InvertedIndex':
        """
        Load a serialized index from disk.

        Args:
            file_path: Full path to the .pkl file.

        Returns:
            InvertedIndex instance.
        """
        with open(file_path, 'rb') as f:
            index = pickle.load(f)

        print(f"[LOADED] Index loaded from {file_path}")
        print(f"[LOADED] {len(index.index)} terms, {index.doc_count} documents")
        return index
