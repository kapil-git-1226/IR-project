"""
build_index.py
--------------
Wires together the full pipeline:
  data_loader -> preprocessor -> indexer

Run from project root:
    uv run python build_index.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_and_process_reuters, save_documents
from preprocessor import TextPreprocessor
from indexer import InvertedIndex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH   = os.path.join(BASE_DIR, 'data', 'raw')
DOCS_PATH       = os.path.join(BASE_DIR, 'data', 'processed', 'documents.json')
INDEX_PATH      = os.path.join(BASE_DIR, 'data', 'index', 'inverted_index.pkl')


# ── Step 1: Load documents ──────────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading and parsing Reuters documents...")
print("=" * 55)

if os.path.exists(DOCS_PATH):
    print(f"[SKIP] documents.json already exists, loading from disk...")
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"[OK] {len(documents)} documents loaded from {DOCS_PATH}")
else:
    documents = load_and_process_reuters(RAW_DATA_PATH)
    save_documents(documents, DOCS_PATH)


# ── Step 2: Build index ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Building inverted index...")
print("=" * 55)

preprocessor = TextPreprocessor()
index = InvertedIndex()
index.build_index(documents, preprocessor)


# ── Step 3: Save index ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Saving index to disk...")
print("=" * 55)

index.save(INDEX_PATH)


# ── Step 4: Verify index ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Verifying index with sample queries...")
print("=" * 55)

test_terms = ['cocoa', 'oil', 'bank', 'stock', 'trade']

for term in test_terms:
    from nltk.stem import PorterStemmer
    stemmed = PorterStemmer().stem(term)
    postings = index.get_postings(stemmed)
    df = index.get_document_frequency(stemmed)
    print(f"  '{term}' (stemmed: '{stemmed}') → {df} documents | sample postings: {postings[:3]}")


# ── Step 5: Load back and confirm ───────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Loading index back from disk...")
print("=" * 55)

loaded_index = InvertedIndex.load(INDEX_PATH)
assert loaded_index.doc_count == index.doc_count, "Doc count mismatch after reload!"
assert len(loaded_index.index) == len(index.index), "Index size mismatch after reload!"
print("[OK] Index reloaded and verified successfully.")

print("\n✅ All steps complete. Index is ready for retrieval.")
