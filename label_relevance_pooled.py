"""
label_relevance_pooled.py
--------------------------
CORRECTED relevance labeling using TREC-style pooling.

Pools the top-N results from TF-IDF, BM25, AND BM25+PRF for each query,
removes duplicates, then asks you to label each unique document once.

This eliminates pooling bias and gives fair scores to all three models.

Run from project root:
    uv run python label_relevance_pooled.py
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

# ── Paths ────────────────────────────────────────────────────────────────────
INDEX_PATH     = os.path.join(BASE_DIR, 'data', 'index', 'inverted_index.pkl')
DOCS_PATH      = os.path.join(BASE_DIR, 'data', 'processed', 'documents.json')
QUERIES_PATH   = os.path.join(BASE_DIR, 'data', 'queries.json')
JUDGMENTS_PATH = os.path.join(BASE_DIR, 'data', 'relevance_judgments.json')

POOL_DEPTH = 10   # How many results to take from EACH model per query

# ── Load resources ───────────────────────────────────────────────────────────
print("Loading index and models...")
index = InvertedIndex.load(INDEX_PATH)
preprocessor = TextPreprocessor()

tfidf = TFIDFRetrieval(index)
bm25  = BM25Retrieval(index)
prf   = PseudoRelevanceFeedback(index, preprocessor)

with open(DOCS_PATH, 'r', encoding='utf-8') as f:
    all_docs = json.load(f)
doc_lookup = {doc['doc_id']: doc['text'] for doc in all_docs}

with open(QUERIES_PATH, 'r') as f:
    queries = json.load(f)

# Load existing judgments to allow resuming
if os.path.exists(JUDGMENTS_PATH):
    with open(JUDGMENTS_PATH, 'r') as f:
        all_judgments = json.load(f)
    print(f"Existing file found — will overwrite with pooled labels.\n")
else:
    all_judgments = {}

# ── Labeling loop ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  POOLED RELEVANCE LABELING TOOL (TREC-style)")
print(f"  Pooling top-{POOL_DEPTH} results from TF-IDF + BM25 + BM25+PRF")
print("  Press Y = Relevant | N = Not Relevant | S = Skip | Q = Quit")
print("=" * 65)

for query_obj in queries:
    qid   = query_obj['query_id']
    qtext = query_obj['text']
    qtype = query_obj['type']
    qdesc = query_obj['description']

    print(f"\n{'─'*65}")
    print(f"  Query ID   : {qid}")
    print(f"  Query Text : \"{qtext}\"")
    print(f"  Type       : {qtype}")
    print(f"  Meaning    : {qdesc}")
    print(f"{'─'*65}")

    # Pool results from all three models
    pool = {}  # doc_id → True (seen)

    r1 = tfidf.search(qtext, preprocessor, top_k=POOL_DEPTH)
    r2 = bm25.search(qtext, preprocessor, top_k=POOL_DEPTH)
    r3, _ = prf.search_with_prf(qtext, bm25, top_k=POOL_DEPTH, feedback_docs=5, expansion_terms=5)

    # Preserve order: BM25 first (highest quality baseline), then others
    ordered_docs = []
    seen = set()
    for doc_id, score in (r2 + r3 + r1):
        if doc_id not in seen:
            ordered_docs.append(doc_id)
            seen.add(doc_id)

    print(f"\n  Pooled {len(ordered_docs)} unique documents from all three models.\n")

    judgments = {}
    skip_query = False

    for rank, doc_id in enumerate(ordered_docs, start=1):
        text = doc_lookup.get(doc_id, "")
        snippet = text[:300].replace('\n', ' ')

        print(f"  Document #{rank}/{len(ordered_docs)} | Doc ID: {doc_id}")
        print(f"  {snippet}...")
        print()

        while True:
            answer = input("  Relevant? [Y/N/S=skip query/Q=quit] : ").strip().lower()
            if answer in ('y', 'n', 's', 'q'):
                break
            print("  Invalid input. Please press Y, N, S, or Q.")

        if answer == 'q':
            print("\nSaving and quitting...")
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
        with open(JUDGMENTS_PATH, 'w') as f:
            json.dump(all_judgments, f, indent=2)
        relevant_count = sum(v for v in judgments.values() if v == 1)
        print(f"\n  ✅ Saved! {relevant_count}/{len(judgments)} documents marked relevant for {qid}.")

print("\n" + "=" * 65)
print("  All queries labeled with pooled judgments!")
print(f"  Saved to: {JUDGMENTS_PATH}")
print("=" * 65)
