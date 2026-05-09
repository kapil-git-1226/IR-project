# Project Context for LLMs
## Ambiguity-Resilient News Retrieval (Reuters-21578)

This file is the authoritative, up-to-date context for any LLM agent working on this repository.
It explains the architecture, execution flow, current implementation status, known gaps, and safe modification strategy.

---

## 1) Project Purpose

This project is an Information Retrieval system for Reuters-21578 news articles.

Core objectives:
- Parse Reuters SGML documents
- Preprocess text (tokenization, stopwords, stemming)
- Build an inverted index
- Retrieve with TF-IDF and BM25
- Apply pseudo-relevance feedback (PRF) query expansion
- Evaluate with pooled relevance judgments (P@5, P@10, Recall, AP, MAP)
- Add enhanced PRF components for ambiguity/noise robustness

---

## 2) Current Repository State (Important)

As of current state:
- Baseline IR pipeline is functional end-to-end.
- `data/raw`, `data/processed/documents.json`, and `data/index/inverted_index.pkl` exist.
- Baseline evaluation script (`experiments/run_experiments.py`) uses **baseline PRF** (`search_with_prf`), not the enhanced method.
- Enhanced modules exist under `src/enhancements`.
- Enhanced PRF entrypoint exists in `src/query_expansion.py` as `search_with_prf_enhanced(...)`.
- Tests folder exists and direct Python execution passes for current test set.

Practical environment notes:
- `uv` may not be installed on some machines; use `python ...` as fallback.
- `pytest` may be missing; tests can be invoked by direct Python import/execution.
- `build_index.py` prints a Unicode arrow character that may fail on Windows cp1252 consoles.

---

## 3) High-Level Architecture

Pipeline:
1. SGML parsing -> `src/data_loader.py`
2. Text preprocessing -> `src/preprocessor.py`
3. Inverted index build/load -> `src/indexer.py` + `build_index.py`
4. Retrieval scoring -> `src/retrieval.py`
5. Query expansion -> `src/query_expansion.py`
6. Metrics -> `evaluation/metrics.py`
7. Experiment runner -> `experiments/run_experiments.py`

Enhancement modules (new):
- `src/enhancements/ambiguity_detector.py`
- `src/enhancements/feedback_scorer.py`
- `src/enhancements/semantic_clusterer.py`

---

## 4) Directory Map

```text
IR-project/
├── data/
│   ├── raw/                       # Reuters SGM files (22 files)
│   ├── processed/documents.json   # Parsed article text
│   ├── index/inverted_index.pkl   # Serialized index artifact
│   ├── queries.json               # 10 query set
│   └── relevance_judgments.json   # pooled labels
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── indexer.py
│   ├── retrieval.py
│   ├── query_expansion.py
│   └── enhancements/
│       ├── __init__.py
│       ├── ambiguity_detector.py
│       ├── feedback_scorer.py
│       └── semantic_clusterer.py
├── evaluation/metrics.py
├── experiments/
│   ├── run_experiments.py
│   └── analysis.md
├── tests/
│   ├── enhancements/
│   │   ├── test_ambiguity_detector.py
│   │   ├── test_feedback_scorer.py
│   │   └── test_semantic_clusterer.py
│   ├── test_integration.py
│   └── test_end_to_end.py
├── demo/
│   ├── index.html
│   ├── static/
│   └── templates/
├── build_index.py
├── search.py
├── label_relevance.py
├── label_relevance_pooled.py
├── MASTER.md
├── MASTER_IMPLEMENTATION_GUIDE.md
└── context.md
```

---

## 5) Core Modules Explained

### `src/data_loader.py`
- Parses Reuters SGML files using BeautifulSoup.
- Extracts `newid`, `title`, and `body`.
- Creates normalized document records:
  - `{"doc_id": "...", "text": "..."}`
- Writes `data/processed/documents.json`.

### `src/preprocessor.py`
- Tokenization via regex
- Stopword removal via NLTK English list
- Porter stemming
- Used consistently for indexing and querying.

### `src/indexer.py`
- Stores:
  - `index[term] -> [(doc_id, tf), ...]`
  - `doc_lengths[doc_id]`
  - `doc_count`, `avg_doc_length`
- Supports save/load with pickle.

### `src/retrieval.py`
- `TFIDFRetrieval.search(...)`
- `BM25Retrieval.search(...)`
- BM25 defaults:
  - `k1 = 1.5`
  - `b = 0.75`

### `src/query_expansion.py`
Class: `PseudoRelevanceFeedback`

Baseline:
- `search_with_prf(...)`
  - initial retrieval
  - select top feedback docs
  - extract expansion terms via corpus-aware filters
  - build deduplicated expanded query
  - rerun retrieval

Enhanced:
- `search_with_prf_enhanced(...)`
  - adaptive ambiguity detection
  - feedback quality filtering
  - semantic clustering term extraction
  - fallback to baseline if doc text is unavailable

Key integration nuance:
- Enhanced method needs document text via `doc_lookup` in PRF constructor.
- If caller instantiates `PseudoRelevanceFeedback(index, preprocessor)` without `doc_lookup`,
  enhanced path cannot resolve doc text and falls back.

### `evaluation/metrics.py`
- `precision_at_k`
- `recall`
- `average_precision`
- `mean_average_precision`

### `experiments/run_experiments.py`
- Loads index, query set, labels
- Runs TF-IDF, BM25, and baseline BM25+PRF
- Prints aggregate + per-query report

---

## 6) Enhancement Modules (Design + Behavior)

### `ambiguity_detector.py`
Goal:
- Determine whether feedback set is semantically coherent.

Method:
- TF-IDF vectorization
- pairwise cosine similarity
- mean similarity = coherence score
- below threshold => ambiguous (conservative expansion)

Returns:
- `is_ambiguous`
- `coherence_score`
- `classification`
- `recommended_params` (feedback docs + expansion terms + weights)

### `feedback_scorer.py`
Goal:
- Filter noisy feedback documents before term extraction.

Method:
- relevance score: cosine(query, doc)
- coherence score: average cosine(doc, other docs)
- quality = alpha * relevance + beta * coherence
- keep docs above threshold
- enforce minimum keep count safety

### `semantic_clusterer.py`
Goal:
- Avoid mixing terms from multiple senses of ambiguous queries.

Method:
- TF-IDF vectorize docs
- similarity graph with threshold edges
- connected components as semantic clusters
- use largest cluster for term extraction

Outputs:
- cluster metadata
- expansion term list (term, score)

---

## 7) Data Contracts

### `data/queries.json`
List of objects:
- `query_id` (e.g., `Q001`)
- `text`
- `type` (`ambiguous` or `specific`)
- `description`

### `data/relevance_judgments.json`
Dictionary:
- key: query id
- value: dictionary of `doc_id -> 0/1`

### `data/processed/documents.json`
List:
- `doc_id` (string)
- `text` (title + body merged and cleaned)

---

## 8) How to Run (Robust Commands)

From project root:

Preferred if `uv` is available:
```bash
uv run python build_index.py
uv run python experiments/run_experiments.py
uv run python search.py
```

Fallback with plain Python:
```bash
python build_index.py
python experiments/run_experiments.py
python search.py
```

Windows encoding-safe run for scripts with unicode output:
```bash
$env:PYTHONIOENCODING='utf-8'; python experiments/run_experiments.py
```

---

## 9) Evaluation Methodology

Model comparison includes:
- TF-IDF baseline
- BM25 baseline
- BM25 + PRF baseline

Metrics:
- P@5
- P@10
- Recall
- AP
- MAP

Labeling protocol:
- Use pooled labeling (`label_relevance_pooled.py`) to reduce model bias.
- Re-labeling changes results, so treat labels as versioned ground truth.

---

## 10) Testing Status

Existing test files:
- `tests/enhancements/test_ambiguity_detector.py`
- `tests/enhancements/test_feedback_scorer.py`
- `tests/enhancements/test_semantic_clusterer.py`
- `tests/test_integration.py`
- `tests/test_end_to_end.py` (currently smoke placeholder)

Current operational note:
- Tests were validated via direct Python invocation in environments where `pytest` is unavailable.

---

## 11) Known Gaps / Next Actions

1. **Enhanced PRF not yet used in experiments**
   - `experiments/run_experiments.py` still calls `search_with_prf(...)`.
   - Next: add an additional model arm for `search_with_prf_enhanced(...)`.

2. **Enhanced PRF caller wiring**
   - For enhanced mode, instantiate:
     - `PseudoRelevanceFeedback(index, preprocessor, doc_lookup=...)`
   - `doc_lookup` should map doc_id -> full document text.

3. **Windows print encoding in `build_index.py`**
   - Unicode arrow in a print line may raise cp1252 encode errors.
   - Replace arrow with ASCII text or set UTF-8 output environment.

4. **README is empty**
   - Add onboarding instructions for setup/run/test.

5. **Test rigor can be improved**
   - Current tests are practical smoke/unit checks.
   - Add deterministic fixture-based assertions for stable CI behavior.

---

## 12) Safe Editing Guidelines for LLM Agents

- Prefer minimal, isolated changes.
- Do not alter query labels unless explicitly requested.
- Do not overwrite `data/relevance_judgments.json` casually.
- Keep baseline behavior intact while adding enhanced alternatives.
- When changing evaluation scripts, preserve old model outputs for comparability.
- Verify with at least:
  - import smoke test
  - enhancement unit tests
  - integration script

---

## 13) Quick Orientation for a New LLM

If you are newly attached to this repo, do this in order:
1. Read this file (`context.md`).
2. Read `src/query_expansion.py` and `experiments/run_experiments.py`.
3. Confirm data artifacts exist in `data/index` and `data/processed`.
4. Run experiments once to establish current baseline.
5. If implementing improvements, add new model arms rather than replacing baseline.

---

## 14) Current Dependencies

From `pyproject.toml`:
- `bs4`
- `lxml`
- `nltk`
- `numpy`
- `scikit-learn`

---

This context is intended to be accurate to the current repository state and should be updated whenever pipeline behavior, evaluation methodology, or enhancement wiring changes.
