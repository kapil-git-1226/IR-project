# Project complete guide

This document explains the **Adaptive Feedback Quality–Aware Query Expansion** system on the **Reuters-21578** collection. It is written for readers with **no prior background** in information retrieval, then connects ideas to **this repository’s code and files**.

---

## 1. What is this project?

### 1.1 Information retrieval (very simple)

**Information retrieval (IR)** means: *given a question (a **query**), find the most useful text documents from a large collection*.

- You can think of it as “search,” but the collection can be **news articles**, not just web pages.
- A **retrieval system** scores every document and returns a **ranked list** (best first).

A common problem: the user’s words do not match the words in relevant articles. For example, someone might search for “**stock market crash**” but a relevant article only says “**Black Monday**” and “**Dow**.” A literal word match can **miss** good documents.

This project works on the **Reuters-21578** newswire: many short articles, financial and world news, real-world vocabulary overlap.

### 1.2 Query expansion (what and why)

**Query expansion** means: *take the user’s short query and turn it into a slightly longer query* by adding **extra terms** that are likely to co-occur with relevant content.

- Goal: **improve recall** (find more of the right documents) and sometimes **precision** (rank the right ones higher).
- Risk: if you add the *wrong* terms, the new query “drifts” to a different topic and **hurts** results.

This project uses **automatic** expansion (the system picks terms), not hand-written synonyms.

### 1.3 Pseudo-relevance feedback (PRF) — what it is

**Pseudo-relevance feedback (PRF)** is a classic automatic expansion method:

1. **Run the first search** with the original query (here: **BM25** over an inverted index).
2. **Assume the top few results are “probably relevant”** (pseudo-relevant), even though we have not asked a human.
3. **Look at the text** of those top documents and **extract** new terms that are statistically strong in that set.
4. **Build an expanded query** = original terms + new terms.
5. **Run a second search** with the expanded query and return the new ranking.

It is “pseudo” because we *pretend* the top results are relevant. That is often true, but **not always**—and that is a main source of failure.

### 1.4 Why PRF matters

- **When it works:** it connects the user’s phrasing to the vocabulary of the corpus (synonyms, entity names, domain terms).
- **When it fails:** the top results are **noisy** or **mixed topics**; expansion **injects bad terms** and the second search becomes worse.

This project is designed to **reduce** those failure modes for news text.

### 1.5 What problems does PRF create?

Typical issues (simplified):

| Problem | What goes wrong |
|--------|------------------|
| **One size fits all** | Every query uses the same number of feedback documents and expansion terms, even when the query is ambiguous. |
| **Noisy top results** | A few off-topic documents in the top-5 can poison the term set. |
| **Sense mixing** | If the top set contains two meanings (e.g. “python” = programming vs snake), expansion can mix both vocabularies. |

The three enhancement modules in `src/enhancements/` target these issues.

---

## 2. The big picture (end-to-end flow)

High-level pipeline used in this project:

```
User query (text)
    ↓
Text preprocessing (tokenize, stopwords, stem)  →  index terms
    ↓
Initial retrieval (BM25)  →  ranked doc IDs + scores
    ↓
Pseudo-relevance set  →  top-N doc IDs treated as feedback
    ↓
(Enhanced path) Load doc texts; run ambiguity / quality / clustering
    ↓
Extract expansion terms  →  build expanded query string
    ↓
Second retrieval (BM25) with expanded query  →  final ranked list
```

- **Initial retrieval** ranks documents that contain query terms (via inverted index + BM25 scoring).
- **Feedback selection** chooses which documents contribute evidence for expansion.
- **Term extraction** picks candidate expansion words from feedback text (or clustered subset).
- **Re-ranking** runs retrieval again; scores can change a lot because the query string changed.

---

## 3. What this codebase fixes (motivation)

### 3.1 Fixed parameters in baseline PRF

Baseline PRF (`search_with_prf`) uses fixed settings such as “top 5 feedback docs, 5 expansion terms” (see `experiments/run_experiments.py`). That is simple and reproducible, but **not adaptive**.

### 3.2 Blind trust in top results

PRF assumes top-ranked docs are good feedback. For ambiguous or broad queries, the top set may be **mixed**, so extracted terms **hurt** the second pass.

### 3.3 Vocabulary mixing (multiple senses)

If feedback documents reflect **different meanings**, expansion terms can combine into an **incoherent** query.

**Our response:** measure coherence, filter low-quality feedback passages, and extract terms from a **dominant semantic cluster** when clustering is enabled.

---

## 4. Our solution: three enhancement modules

All live under `src/enhancements/`.

### 4.1 Module 1 — Ambiguity detection (`ambiguity_detector.py`)

**Problem addressed:** feedback sets that actually contain **multiple topics** (ambiguous pseudo-relevance).

**Core idea:** measure **average pairwise cosine similarity** between TF–IDF vectors of feedback documents.

- **Coherence score** ≈ how similar the feedback passages are to each other.
- Default rule in code: `AmbiguityDetector(similarity_threshold=0.4)` — if coherence **≥** threshold → treat as **SPECIFIC**; else **AMBIGUOUS**.
- **Recommended parameters** switch between conservative vs aggressive counts for feedback docs and expansion terms (see class defaults).

**Code:** `AmbiguityDetector.detect_ambiguity(feedback_docs)`.

**Note:** the **web demo** (`demo/server.py`) uses **different** numeric thresholds (tighter: `0.04`, `0.08`, `0.16`) for visualization; the **library defaults** are defined in the enhancement modules and `search_with_prf_enhanced`.

### 4.2 Module 2 — Quality filtering (`feedback_scorer.py`)

**Problem addressed:** some pseudo-relevant documents are **weak** (off-topic or inconsistent with others).

**Core idea:** for each feedback doc, compute:

- **Relevance:** cosine similarity between query vector and doc vector.
- **Coherence:** average similarity between this doc and *other* feedback docs.

Combine into **quality** = `0.6 × relevance + 0.4 × coherence` (default weights). Drop docs below **quality_threshold** (default `0.5`), with a safety rule to keep at least two docs when possible.

**Code:** `FeedbackQualityScorer.score_feedback_documents(query, feedback_docs)`.

### 4.3 Module 3 — Semantic clustering (`semantic_clusterer.py`)

**Problem addressed:** mixed senses / facets inside the feedback set.

**Core idea:** build a similarity graph on feedback docs (edge if cosine similarity ≥ threshold, default `0.5`), take **connected components** as clusters, then extract expansion terms from the **largest** cluster (implementation details in `cluster_and_expand`).

**Code:** `SemanticFeedbackClusterer.cluster_feedback` / `cluster_and_expand`.

---

## 5. System architecture (where code lives)

```
IR-project/
├── data/
│   ├── raw/                     # Reuters .sgm sources (not always in git)
│   ├── processed/documents.json # Parsed articles (doc_id + text)
│   ├── index/inverted_index.pkl
│   ├── queries.json             # Evaluation queries + type tags
│   └── relevance_judgments.json
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── indexer.py
│   ├── retrieval.py             # TF-IDF & BM25
│   ├── query_expansion.py       # Baseline PRF + enhanced PRF API
│   └── enhancements/
│       ├── ambiguity_detector.py
│       ├── feedback_scorer.py
│       └── semantic_clusterer.py
├── evaluation/metrics.py        # P@k, recall, AP, MAP helpers
├── experiments/
│   └── run_experiments.py       # Compares models on labeled queries
├── demo/
│   ├── server.py                # Serves UI + /analyze API
│   ├── static/app.js
│   └── static/styles.css
├── build_index.py
└── search.py
```

**Important:** there is **no** separate `frontend/` folder; the UI is under **`demo/`**.

---

## 6. Data flow (trace one query)

### 6.1 Preprocessing

File: `src/preprocessor.py`

- Lowercasing, tokenization, stopword removal, **Porter stemming** (so “running” and “runs” may match).

### 6.2 Indexing

Files: `src/indexer.py`, `build_index.py`

- Inverted index maps **stemmed term → list of (doc_id, term frequency)**.
- BM25 uses document lengths and corpus statistics.

### 6.3 Baseline PRF (`search_with_prf`)

File: `src/query_expansion.py`

1. BM25 search → top `feedback_docs` IDs.
2. Extract expansion terms from those docs using corpus + feedback constraints (`_extract_expansion_terms`).
3. Concatenate unique terms → expanded query string.
4. BM25 search again → final list.

### 6.4 Enhanced PRF API (`search_with_prf_enhanced`)

Same file: runs **ambiguity → quality filter → clustering** when document text is available via `doc_lookup`.

If texts are missing, it **falls back** to baseline PRF.

### 6.5 Demo pipeline (`demo/server.py`)

The HTTP demo calls `compute_pipeline`, which wires the three modules and BM25; it can use **corpus-derived feedback** when the user leaves the feedback box empty (BM25 top hits as pseudo-relevance).

---

## 7. Evaluation vs “enhanced” naming (read this carefully)

File: `experiments/run_experiments.py`

The row labeled **“BM25 + Enhanced PRF”** is **not** calling `search_with_prf_enhanced`.

It implements an **oracle adaptive rule** using `queries.json`:

- If `type == "specific"` → run **baseline PRF** (`search_with_prf`).
- Else → run **BM25 only** (no PRF).

That design uses **human query-type labels** shipping with the benchmark. It is useful for reporting, but it is **not** the same thing as the full `search_with_prf_enhanced` pipeline.

The **full enhancement stack** is available in code and in the **demo**, but the table row name can be misleading unless you document this distinction (this section does).

---

## 8. Evaluation metrics (what the numbers mean)

File: `evaluation/metrics.py`

- **Precision@k:** among the top *k* retrieved docs, what fraction are relevant?
- **Recall:** what fraction of *all* relevant docs appear anywhere in the retrieved list (implementation uses set overlap — see code).
- **Average precision (AP):** averages precision at each rank where a relevant doc appears.
- **MAP:** mean of AP across queries.

Run:

```bash
python experiments/run_experiments.py
```

Exact MAP values **depend on your index and judgments**; do not treat example numbers in external prompts as guaranteed.

---

## 9. How to run the project

From the project root (examples):

```bash
python build_index.py
python experiments/run_experiments.py
python search.py
```

**Demo UI (optional):**

```bash
python demo/server.py
```

Then open the URL printed in the terminal (this repo defaults to **port 8000** in `demo/server.py`). The browser loads `demo/index.html`, which imports `demo/static/app.js`.

---

## 10. Novelty 2 (outcome prediction) — status in this repository

Some research notes discuss training a classifier to **predict whether PRF will help before expanding**. That is **not part of the committed codebase** at the time this guide was generated (after resetting to the Novelty-1 snapshot). If you reintroduce it, document it as a separate module and evaluation script.

---

## Quick glossary

| Term | Meaning |
|------|---------|
| **Corpus** | The whole document collection (here: Reuters articles). |
| **Inverted index** | Term → postings lists for fast lookup. |
| **BM25** | A strong classical ranking function for sparse queries. |
| **PRF** | Use top retrieval results as fake relevance feedback for expansion. |
| **MAP** | A standard aggregate quality measure across queries. |

---

*End of `PROJECT_COMPLETE_GUIDE.md`*
