# Literature gaps and how this project addresses them

This note compares **classic PRF literature** (Rocchio-style assumptions, blind feedback, PRF-for-news) with **what this repository implements**, and frames contributions honestly—including where evaluation scripts use **oracle query-type labels** rather than automatic ambiguity prediction alone.

---

## 1. Classic PRF (Rocchio-style)

**Main ideas**

- Treat top-ranked documents as **pseudo-positive** examples.
- Reformulate the query vector toward the centroid of those documents (Rocchio), or (in **Okapi/BM25-oriented** pipelines) extract salient terms from top passages and **append** them to the query string.

**Strengths**

- Strong baseline when top results are **clean** and **topic-coherent**.

**Limitations (documented in textbooks and papers)**

| Gap | Short explanation |
|-----|-------------------|
| **Fixed hyperparameters** | Same feedback depth / expansion width for every query. |
| **Noise sensitivity** | A few bad documents in the feedback set poison expansion. |
| **Ambiguity** | Polysemy or broad queries yield mixed feedback semantics. |

---

## 2. PRF on news collections (Reuters-style corpora)

News articles are short; vocabulary is sparse; entity names and synonyms vary. PRF can **bridge vocabulary**, but news clusters also contain **near-duplicates** and **tangentially related** stories, which increases **noise** in pseudo-relevance sets.

**Literature gap:** methods often report aggregate MAP gains without explicitly separating **ambiguous vs specific** query regimes at expansion time.

---

## 3. What this codebase adds (three mechanisms)

| Literature gap | Mechanism in repo | Primary file |
|----------------|-------------------|--------------|
| Non-adaptive feedback/expansion sizes | Coherence-based routing of **feedback_doc** and **expansion_term** counts | `src/enhancements/ambiguity_detector.py` |
| Noisy pseudo-relevant passages | Weighted **relevance + coherence** score per doc; threshold filtering | `src/enhancements/feedback_scorer.py` |
| Mixed senses / multi-topic feedback | Graph clustering on cosine similarity; extract terms from **dominant cluster** | `src/enhancements/semantic_clusterer.py` |

These map cleanly onto known PRF failure modes: **parameter rigidity**, **top-result unreliability**, and **semantic drift**.

---

## 4. Critical distinction: experiment row vs full enhanced pipeline

**Full pipeline API:** `PseudoRelevanceFeedback.search_with_prf_enhanced` in `src/query_expansion.py` chains ambiguity → quality → clustering when document texts exist.

**Benchmark script (`experiments/run_experiments.py`):** the row named **“BM25 + Enhanced PRF”** uses **`queries.json` type labels**:

- **specific** → baseline PRF (`search_with_prf`)
- **ambiguous** → BM25 only

That is **oracle routing** by human-assigned query type. It answers “what if we knew when **not** to fire PRF?”—not “does `search_with_prf_enhanced` beat baseline without labels?”

**Honest contribution framing:**

- **Strong:** improvements from ambiguity-aware parameterization, quality filtering, and clustering when evaluated **fairly** (same retrieval backbone, document access, and judgments).
- **Needs care:** any claim that “enhanced PRF” won because of the benchmark table must cite **which branch** of the experiment script was used.

---

## 5. Summary comparison table (papers vs this project)

| Topic | Typical paper assumption | This project |
|-------|--------------------------|--------------|
| Feedback set quality | Implicitly good if BM25 ranks well | Explicit **per-doc quality** + removal rule |
| Ambiguity | Often ignored or handled via query reformulation elsewhere | **Coherence score** drives conservative vs aggressive expansion |
| Semantic consistency | Single centroid / bag-of-words overlap | **Cluster feedback**, expand from largest coherent subset |
| Evaluation | MAP / NDCG on static collections | MAP helpers + labeled Reuters queries in `data/queries.json` |

---

## 6. Planned / external ideas not guaranteed in-tree

Some write-ups mention **PRF outcome prediction** (train a classifier on query-level features to decide whether to expand). That idea is **not assumed present** in the repository snapshot this document accompanies; treat it as **future work** unless you verify matching Python modules and training scripts exist.

---

## 7. How to describe contributions (thesis-ready bullets)

1. **Adaptive pseudo-relevance:** coherence among feedback passages adjusts expansion aggressiveness before term extraction.
2. **Quality-aware filtering:** combines query–document relevance with intra-feedback coherence to reduce poison feedback.
3. **Semantic clustering:** restricts expansion term mining to a dominant coherent subset when graph connectivity separates facets.
4. **Evaluation hygiene:** explicitly separate **oracle query-type routing** in `run_experiments.py` from the **library enhanced PRF** entry point to avoid overstating automation.

---

## References (conceptual—not exhaustive)

- Rocchio-style relevance feedback (vector reformulation).
- Blind / pseudo-relevance feedback surveys describing noise and ambiguity risks.
- Robertson & Walker BM25 family (sparse retrieval backbone).
- Reuters-21578 as a standard IR test collection.

---

*End of `LITERATURE_GAPS_AND_SOLUTIONS.md`*
