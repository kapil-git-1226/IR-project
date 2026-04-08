"""
search.py
---------
Interactive search script for the Reuters IR system.
Shows results from TF-IDF, BM25, and BM25+PRF side by side.

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
from query_expansion import PseudoRelevanceFeedback


# ── Load resources ──────────────────────────────────────────────────────────
INDEX_PATH = os.path.join(BASE_DIR, 'data', 'index', 'inverted_index.pkl')
DOCS_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'documents.json')

print("Loading index and documents...")
index = InvertedIndex.load(INDEX_PATH)
preprocessor = TextPreprocessor()

tfidf = TFIDFRetrieval(index)
bm25  = BM25Retrieval(index)
prf   = PseudoRelevanceFeedback(index, preprocessor)

with open(DOCS_PATH, 'r', encoding='utf-8') as f:
    all_docs = json.load(f)
doc_lookup = {doc['doc_id']: doc['text'] for doc in all_docs}

print(f"Ready! {index.doc_count:,} documents | {len(index.index):,} unique terms")
print("Type 'exit' to quit.\n")


# ── Display helper ──────────────────────────────────────────────────────────
def display_results(results, model_name, doc_lookup, top_k, extra_info=None):
    print(f"\n{'─'*60}")
    print(f"  📌 {model_name} — Top {top_k} Results")
    if extra_info:
        print(f"  🔎 Expanded query: \"{extra_info}\"")
    print(f"{'─'*60}")

    if not results:
        print("  No results found.")
        return

    for rank, (doc_id, score) in enumerate(results, start=1):
        text = doc_lookup.get(doc_id, "")
        snippet = text[:200].replace('\n', ' ') + "..."
        print(f"\n  #{rank} | Doc {doc_id} | Score: {score:.4f}")
        print(f"  {snippet}")
    print()


# ── Interactive search loop ─────────────────────────────────────────────────
TOP_K = 5

while True:
    query = input("🔍 Enter query (or 'exit'): ").strip()

    if query.lower() == 'exit':
        print("Goodbye!")
        break
    if not query:
        continue

    print(f"\n{'='*60}")
    print(f"  Query: \"{query}\"")
    print(f"{'='*60}")

    # TF-IDF baseline
    tfidf_results = tfidf.search(query, preprocessor, top_k=TOP_K)
    display_results(tfidf_results, "TF-IDF (Baseline)", doc_lookup, TOP_K)

    # BM25 baseline
    bm25_results = bm25.search(query, preprocessor, top_k=TOP_K)
    display_results(bm25_results, "BM25 (Baseline)", doc_lookup, TOP_K)

    # BM25 + PRF
    prf_results, expanded_query = prf.search_with_prf(
        query,
        bm25,
        top_k=TOP_K,
        feedback_docs=5,      # Use more docs to get broader context
        expansion_terms=5      # Add fewer expansion terms to reduce drift
    )
    display_results(prf_results, "BM25 + PRF (Expanded)", doc_lookup, TOP_K, extra_info=expanded_query)
