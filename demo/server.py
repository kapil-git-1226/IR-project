import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from indexer import InvertedIndex
from preprocessor import TextPreprocessor
from retrieval import BM25Retrieval
from enhancements.ambiguity_detector import AmbiguityDetector
from enhancements.feedback_scorer import FeedbackQualityScorer
from enhancements.semantic_clusterer import SemanticFeedbackClusterer


INDEX_PATH = PROJECT_ROOT / "data" / "index" / "inverted_index.pkl"
DOCS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.json"
EXAMPLES_PATH = BASE_DIR / "examples.json"

print("Loading IR resources...")
INDEX = InvertedIndex.load(str(INDEX_PATH))
PREPROCESSOR = TextPreprocessor()
BM25 = BM25Retrieval(INDEX)

with open(DOCS_PATH, "r", encoding="utf-8") as f:
    ALL_DOCS = json.load(f)
DOC_LOOKUP = {doc["doc_id"]: doc["text"] for doc in ALL_DOCS}


def unique_terms(sequence):
    seen = set()
    out = []
    for t in sequence:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _preview(text: str, n: int = 260) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def compute_pipeline(query, feedback_docs):
    detector = AmbiguityDetector(similarity_threshold=0.04)
    scorer = FeedbackQualityScorer(quality_threshold=0.08)
    clusterer = SemanticFeedbackClusterer(similarity_threshold=0.16)

    ambiguity = detector.detect_ambiguity(feedback_docs)
    expansion_count = ambiguity["recommended_params"]["expansion_terms"]

    quality = scorer.score_feedback_documents(query, feedback_docs)
    filtered_docs = quality["filtered_docs"]
    if not filtered_docs:
        filtered_docs = feedback_docs[:]

    cluster = clusterer.cluster_and_expand(
        filtered_docs, query, num_expansion_terms=expansion_count
    )
    expansion_terms = [term for term, _ in cluster["expansion_terms"]]

    query_terms = PREPROCESSOR.preprocess(query)
    expanded_query = " ".join(unique_terms(query_terms + expansion_terms))
    if not expanded_query:
        expanded_query = query

    search_results = BM25.search(expanded_query, PREPROCESSOR, top_k=5)

    return {
        "prf_input_snippets": [_preview(d) for d in feedback_docs],
        "prf_after_quality_snippets": [_preview(d) for d in filtered_docs],
        "coherence": ambiguity["coherence_score"],
        "classification": ambiguity["classification"],
        "is_ambiguous": ambiguity["is_ambiguous"],
        "kept_count": quality.get("num_kept", 0),
        "filtered_count": quality.get("num_filtered", 0),
        "avg_quality": quality.get("avg_quality_kept", 0.0),
        "num_clusters": cluster["cluster_result"]["num_clusters"],
        "largest_cluster": cluster["cluster_result"]["docs_in_largest"],
        "coverage": cluster["cluster_result"]["coverage"],
        "expansion_terms": expansion_terms,
        "expanded_query": expanded_query,
        "top_results": [
            {
                "doc_id": doc_id,
                "score": score,
                "snippet": DOC_LOOKUP.get(doc_id, "")[:360].replace("\n", " "),
            }
            for doc_id, score in search_results
        ],
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def _send_json(self, payload, status=200):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/examples":
            if EXAMPLES_PATH.exists():
                with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"examples": []}
            self._send_json(data)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/analyze":
            self._send_json({"error": "Not found"}, status=404)
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_len)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        query = (payload.get("query") or "").strip()
        if not query:
            self._send_json({"error": "Query is required"}, status=400)
            return

        feedback_docs = payload.get("feedback_docs") or []
        feedback_docs = [d.strip() for d in feedback_docs if isinstance(d, str) and d.strip()]

        # If user does not provide docs, derive feedback from corpus using BM25.
        if not feedback_docs:
            base_results = BM25.search(query, PREPROCESSOR, top_k=5)
            feedback_docs = [
                DOC_LOOKUP.get(doc_id, "")
                for doc_id, _ in base_results
                if DOC_LOOKUP.get(doc_id, "")
            ]

        if not feedback_docs:
            self._send_json({"error": "No feedback documents available"}, status=400)
            return

        result = compute_pipeline(query, feedback_docs)
        self._send_json({"ok": True, "query": query, "feedback_docs": feedback_docs, "result": result})


def run(port=8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Demo server running: http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()

