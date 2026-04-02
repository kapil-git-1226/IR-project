"""
search.py
---------
Interactive search script for the Reuters IR system.
Loads the inverted index and lets you type queries to see
ranked results from both TF-IDF and BM25 side by side.

Run from project root:
    uv run python search.py
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from indexer import InvertedIndex
from preprocessor import TextPreprocessor
from retrieval import TFIDFRetrieval, BM25Retrieval


# ── Load resources ─────────────────────────────────────────────────────────
INDEX_PATH = os.path.join(BASE_DIR, 'data', 'index', 'inverted_index.pkl')
DOCS_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'documents.json')

print("Loading index and documents...")
index = InvertedIndex.load(INDEX_PATH)
preprocessor = TextPreprocessor()

tfidf = TFIDFRetrieval(index)
bm25  = BM25Retrieval(index)

# Load documents into a dict for fast lookup by doc_id
with open(DOCS_PATH, 'r', encoding='utf-8') as f:
    all_docs = json.load(f)
doc_lookup = {doc['doc_id']: doc['text'] for doc in all_docs}

print(f"Ready! Index has {index.doc_count:,} documents and {len(index.index):,} unique terms.")
print("Type 'exit' to quit.\n")


# ── Interactive search loop ────────────────────────────────────────────────
def display_results(results, model_name, doc_lookup, top_k):
    print(f"\n{'='*60}")
    print(f"  {model_name} — Top {top_k} Results")
    print(f"{'='*60}")

    if not results:
        print("  No results found.")
        return

    for rank, (doc_id, score) in enumerate(results, start=1):
        text = doc_lookup.get(doc_id, "")
        snippet = text[:180].replace('\n', ' ') + "..."
        print(f"\n  #{rank} | Doc ID: {doc_id} | Score: {score:.4f}")
        print(f"  {snippet}")

    print()


while True:
    query = input("🔍 Enter query: ").strip()

    if query.lower() == 'exit':
        print("Goodbye!")
        break

    if not query:
        continue

    top_k = 5

    tfidf_results = tfidf.search(query, preprocessor, top_k=top_k)
    bm25_results  = bm25.search(query, preprocessor, top_k=top_k)

    display_results(tfidf_results, "TF-IDF", doc_lookup, top_k)
    display_results(bm25_results,  "BM25",   doc_lookup, top_k)
