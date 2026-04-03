"""
run_experiments.py
------------------
Evaluates and compares TF-IDF, BM25, and BM25+PRF models across
all labelled test queries. Computes P@5, P@10, Recall, AP and MAP
for each model and prints a clean comparison report.

Run from project root:
    uv run python run_experiments.py
"""

import os
import sys
import json

# experiments/ is one level below the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, os.path.join(BASE_DIR, 'evaluation'))

from indexer import InvertedIndex
from preprocessor import TextPreprocessor
from retrieval import TFIDFRetrieval, BM25Retrieval
from query_expansion import PseudoRelevanceFeedback
from metrics import precision_at_k, recall, average_precision, mean_average_precision

# ── Paths ───────────────────────────────────────────────────────────────────
INDEX_PATH     = os.path.join(BASE_DIR, 'data', 'index', 'inverted_index.pkl')
QUERIES_PATH   = os.path.join(BASE_DIR, 'data', 'queries.json')
JUDGMENTS_PATH = os.path.join(BASE_DIR, 'data', 'relevance_judgments.json')

TOP_K          = 15   # How many results to retrieve & evaluate

# ── Load resources ──────────────────────────────────────────────────────────
print("Loading index and models...")
index = InvertedIndex.load(INDEX_PATH)
preprocessor = TextPreprocessor()

tfidf = TFIDFRetrieval(index)
bm25  = BM25Retrieval(index)
prf   = PseudoRelevanceFeedback(index, preprocessor)

with open(QUERIES_PATH, 'r') as f:
    queries = json.load(f)

with open(JUDGMENTS_PATH, 'r') as f:
    all_judgments = json.load(f)

print(f"Loaded {len(queries)} queries | {len(all_judgments)} labeled\n")


# ── Experiment runner ────────────────────────────────────────────────────────
def run_model(model_name, search_fn, queries, all_judgments, top_k):
    """
    Run a model over all queries and return per-query and aggregate metrics.
    search_fn(query_text) must return a List[(doc_id, score)].
    """
    ap_scores   = []
    p5_scores   = []
    p10_scores  = []
    recall_scores = []
    per_query   = []

    for q in queries:
        qid   = q['query_id']
        qtext = q['text']

        if qid not in all_judgments:
            continue

        judgments    = all_judgments[qid]
        relevant_set = {doc_id for doc_id, label in judgments.items() if label == 1}

        results      = search_fn(qtext)
        retrieved    = [doc_id for doc_id, _ in results]

        ap  = average_precision(retrieved, relevant_set)
        p5  = precision_at_k(retrieved, relevant_set, k=5)
        p10 = precision_at_k(retrieved, relevant_set, k=10)
        rec = recall(retrieved, relevant_set)

        ap_scores.append(ap)
        p5_scores.append(p5)
        p10_scores.append(p10)
        recall_scores.append(rec)

        per_query.append({
            'qid': qid, 'text': qtext, 'type': q['type'],
            'AP': ap, 'P@5': p5, 'P@10': p10, 'Recall': rec
        })

    return {
        'model':   model_name,
        'MAP':     mean_average_precision(ap_scores),
        'P@5':     sum(p5_scores)  / len(p5_scores)  if p5_scores  else 0,
        'P@10':    sum(p10_scores) / len(p10_scores) if p10_scores else 0,
        'Recall':  sum(recall_scores) / len(recall_scores) if recall_scores else 0,
        'per_query': per_query
    }


# ── Run all three models ─────────────────────────────────────────────────────
results_tfidf = run_model(
    "TF-IDF",
    lambda q: tfidf.search(q, preprocessor, top_k=TOP_K),
    queries, all_judgments, TOP_K
)

results_bm25 = run_model(
    "BM25",
    lambda q: bm25.search(q, preprocessor, top_k=TOP_K),
    queries, all_judgments, TOP_K
)

results_prf = run_model(
    "BM25 + PRF",
    lambda q: prf.search_with_prf(q, bm25, top_k=TOP_K, feedback_docs=5, expansion_terms=5)[0],
    queries, all_judgments, TOP_K
)


# ── Print report ─────────────────────────────────────────────────────────────
models = [results_tfidf, results_bm25, results_prf]

print("=" * 65)
print("  EVALUATION REPORT — Ambiguity-Aware News Retrieval System")
print("=" * 65)

# Aggregate comparison table
print(f"\n{'Model':<18} {'MAP':>8} {'P@5':>8} {'P@10':>8} {'Recall':>8}")
print("─" * 50)
for m in models:
    print(f"  {m['model']:<16} {m['MAP']:>8.4f} {m['P@5']:>8.4f} {m['P@10']:>8.4f} {m['Recall']:>8.4f}")

# Per-query breakdown
print(f"\n{'─'*65}")
print("  PER-QUERY BREAKDOWN")
print(f"{'─'*65}")

for q in queries:
    qid = q['query_id']
    if qid not in all_judgments:
        continue

    print(f"\n  {qid} | \"{q['text']}\" [{q['type']}]")
    print(f"  {'Model':<16} {'AP':>8} {'P@5':>8} {'P@10':>8} {'Recall':>8}")
    print(f"  {'─'*48}")

    for m in models:
        row = next((r for r in m['per_query'] if r['qid'] == qid), None)
        if row:
            print(f"  {m['model']:<16} {row['AP']:>8.4f} {row['P@5']:>8.4f} {row['P@10']:>8.4f} {row['Recall']:>8.4f}")

# PRF improvement summary
print(f"\n{'─'*65}")
print("  PRF IMPROVEMENT OVER BM25 BASELINE")
print(f"{'─'*65}")
map_gain    = results_prf['MAP']   - results_bm25['MAP']
p5_gain     = results_prf['P@5']   - results_bm25['P@5']
p10_gain    = results_prf['P@10']  - results_bm25['P@10']
recall_gain = results_prf['Recall']- results_bm25['Recall']

def fmt(val):
    arrow = "▲" if val >= 0 else "▼"
    return f"{arrow} {abs(val):.4f}"

print(f"\n  MAP    : {fmt(map_gain)}")
print(f"  P@5    : {fmt(p5_gain)}")
print(f"  P@10   : {fmt(p10_gain)}")
print(f"  Recall : {fmt(recall_gain)}")

print(f"\n{'='*65}")
print("  Experiment complete.")
print(f"{'='*65}\n")
