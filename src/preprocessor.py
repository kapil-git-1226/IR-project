import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List

# Download required NLTK data (safe to call multiple times)
nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> List[str]:
        """Lowercase and extract word tokens using regex."""
        text = text.lower()
        return re.findall(r'\b[a-z0-9]+\b', text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords and single-character tokens."""
        return [
            t for t in tokens
            if t not in self.stop_words and len(t) > 1
        ]

    def stem(self, tokens: List[str]) -> List[str]:
        """Apply Porter Stemmer to each token."""
        return [self.stemmer.stem(t) for t in tokens]

    def preprocess(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline: tokenize → remove stopwords → stem.

        Args:
            text: Raw input string.

        Returns:
            List of stemmed, filtered tokens.
        """
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        return tokens


if __name__ == '__main__':
    import json
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_path = os.path.join(BASE_DIR, 'data', 'processed', 'documents.json')

    with open(doc_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    preprocessor = TextPreprocessor()

    print(f"Total documents loaded: {len(documents)}")
    print(f"Testing preprocessor on first 3 documents...\n")

    for doc in documents[:3]:
        tokens = preprocessor.preprocess(doc['text'])
        print(f"Doc ID : {doc['doc_id']}")
        print(f"Raw    : {doc['text'][:120]}...")
        print(f"Tokens : {tokens[:15]}")
        print(f"Count  : {len(tokens)} tokens")
        print()
