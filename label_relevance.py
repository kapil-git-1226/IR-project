"""
label_relevance.py
------------------
Interactive script to manually create relevance_judgments.json.

For each query, it runs BM25 and shows you the top-20 documents.
You simply press Y (relevant) or N (not relevant) for each one.
Results are auto-saved to data/relevance_judgments.json.

Run from project root:
    uv run python label_relevance.py
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from indexer import InvertedIndex
from preprocessor import TextPreprocessor
from retrieval import BM25Retrieval

# ── Paths ───────────────────────────────────────────────────────────────────
INDEX_PATH    = os.path.join(BASE_DIR, 'data', 'index', 'inverted_index.pkl')
DOCS_PATH     = os.path.join(BASE_DIR, 'data', 'processed', 'documents.json')
QUERIES_PATH  = os.path.join(BASE_DIR, 'data', 'queries.json')
JUDGMENTS_PATH = os.path.join(BASE_DIR, 'data', 'relevance_judgments.json')

DOCS_TO_LABEL = 15   # How many docs to show per query

# ── Load everything ─────────────────────────────────────────────────────────
print("Loading index...")
index = InvertedIndex.load(INDEX_PATH)
preprocessor = TextPreprocessor()
bm25 = BM25Retrieval(index)

with open(DOCS_PATH, 'r', encoding='utf-8') as f:
    all_docs = json.load(f)
doc_lookup = {doc['doc_id']: doc['text'] for doc in all_docs}

with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

# Load existing judgments if resuming
if os.path.exists(JUDGMENTS_PATH):
    with open(JUDGMENTS_PATH, 'r') as f:
        all_judgments = json.load(f)
    print(f"Resuming — {len(all_judgments)} queries already labeled.\n")
else:
    all_judgments = {}

# ── Labeling loop ────────────────────────────────────────────────────────────
print("=" * 65)
print("  RELEVANCE LABELING TOOL")
print("  Press Y = Relevant | N = Not Relevant | S = Skip query | Q = Quit")
print("=" * 65)

for query_obj in queries:
    qid   = query_obj['query_id']
    qtext = query_obj['text']
    qtype = query_obj['type']
    qdesc = query_obj['description']

    if qid in all_judgments:
        print(f"\n[SKIP] {qid} already labeled. Skipping...")
        continue

    print(f"\n{'─'*65}")
    print(f"  Query ID   : {qid}")
    print(f"  Query Text : \"{qtext}\"")
    print(f"  Type       : {qtype}")
    print(f"  Meaning    : {qdesc}")
    print(f"{'─'*65}")

    results = bm25.search(qtext, preprocessor, top_k=DOCS_TO_LABEL)
    if not results:
        print("  No results found for this query. Skipping.")
        all_judgments[qid] = {}
        continue

    judgments = {}
    skip_query = False

    for rank, (doc_id, score) in enumerate(results, start=1):
        text = doc_lookup.get(doc_id, "")
        snippet = text[:300].replace('\n', ' ')

        print(f"\n  Result #{rank} | Doc ID: {doc_id} | BM25 Score: {score:.4f}")
        print(f"  {snippet}...")
        print()

        while True:
            answer = input("  Relevant? [Y/N/S=skip query/Q=quit] : ").strip().lower()
            if answer in ('y', 'n', 's', 'q'):
                break
            print("  Invalid input. Please press Y, N, S, or Q.")

        if answer == 'q':
            print("\nSaving progress and quitting...")
            all_judgments[qid] = judgments
            with open(JUDGMENTS_PATH, 'w') as f:
                json.dump(all_judgments, f, indent=2)
            sys.exit(0)
        elif answer == 's':
            skip_query = True
            break
        else:
            judgments[doc_id] = 1 if answer == 'y' else 0

    if not skip_query:
        all_judgments[qid] = judgments
        # Save after every query so progress is never lost
        with open(JUDGMENTS_PATH, 'w') as f:
            json.dump(all_judgments, f, indent=2)
        relevant_count = sum(v for v in judgments.values() if v == 1)
        print(f"\n  ✅ Saved! {relevant_count}/{len(judgments)} documents marked relevant for {qid}.")

print("\n" + "=" * 65)
print("  All queries labeled!")
print(f"  Judgments saved to: {JUDGMENTS_PATH}")
print("=" * 65)
