# CURSOR PROMPT: GENERATE COMPLETE PROJECT DOCUMENTATION

You are creating comprehensive documentation for an IR project. The user has ZERO background and needs complete understanding of everything.

---

## TASK 1: CREATE COMPLETE PROJECT OVERVIEW MD

**File to create:** `PROJECT_COMPLETE_GUIDE.md`

**Structure:**

### SECTION 1: WHAT IS THIS PROJECT? (For Complete Beginners)

Start with absolute basics. Assume reader knows nothing.

Content needed:
- What is Information Retrieval? (2 paragraphs, ELI5)
- What is Query Expansion? (definition + why it matters)
- What is Pseudo-Relevance Feedback? (definition + how it works, step-by-step)
- Why is PRF important in IR?
- What problems does PRF solve? What problems does it create?

Example structure:
```
## Information Retrieval Basics

IR is finding relevant documents for a user query.

User: "machine learning"
System: searches database, returns 10 articles about ML

Problem: User used term "machine learning", but some relevant articles 
use "deep learning" or "neural networks" instead. System misses them.

Solution: Query Expansion - automatically add related terms to query.
New query: "machine learning" + "deep learning" + "neural networks"
Result: Find more relevant articles
```

### SECTION 2: THE BIG PICTURE (End-to-End System Flow)

Diagram (ASCII or detailed description) showing:
```
User Query
    ↓
[Initial Retrieval] → BM25 Search → Top-5 Documents
    ↓
[Feedback Selection] → Are these docs relevant?
    ↓
[Term Extraction] → What terms appear in these docs?
    ↓
[Query Expansion] → Add terms to original query
    ↓
[Re-Ranking] → Search again with expanded query
    ↓
Final Results to User
```

Explain each step in 2-3 sentences.

### SECTION 3: THE PROBLEM (What We're Fixing)

Explain 3 core problems in standard PRF:

**Problem 1: Fixed Parameters**
- Standard PRF uses same expansion for all queries
- Example: "apple" (ambiguous) vs "stock market crash" (specific)
- Both get same 5 feedback docs + 5 expansion terms
- Result: "apple" expands to mixed fruit+Apple Inc vocabulary (bad)

**Problem 2: Blind Trust in Feedback**
- System assumes all top-5 documents are relevant
- But 40% of queries have noisy documents in top-5
- Noisy docs inject wrong terms
- Result: Expansion actually hurts retrieval

**Problem 3: Vocabulary Mixing**
- When "apple" returns docs about fruit AND Apple Inc
- Expansion extracts terms from BOTH
- Result: expanded query is confused, retrieves fruit docs when user wanted Apple Inc

**How Literature Identifies These:**
- Abe et al. (2505.12694): LLM expansion fails on ambiguous queries (35% drop)
- Li et al. (2205.05888): 38.9% of queries harmed by noisy feedback
- Sanderson (word sense disambiguation): Multiple senses confuse expansion

### SECTION 4: OUR SOLUTION (3 Modules)

For each module, explain:
- Problem it solves
- How it works (step-by-step)
- Code location
- Example walkthrough

**Module 1: Ambiguity Detection**
- Problem: Can't distinguish "apple" (ambiguous) from "market crash" (specific)
- Solution: Measure feedback document similarity
  * If docs are diverse (low coherence) → query is ambiguous
  * If docs are similar (high coherence) → query is specific
- Formula: coherence_score = avg_cosine_similarity(feedback_docs)
- Threshold: 0.4 (below = ambiguous, above = specific)
- Adaptive response:
  * Ambiguous: conservative expansion (3 docs, 3 terms)
  * Specific: aggressive expansion (7 docs, 8 terms)

**Module 2: Quality Filtering**
- Problem: Noisy docs in feedback hurt expansion
- Solution: Score each feedback doc on 2 metrics
  * Relevance: how similar to query?
  * Coherence: how similar to other feedback docs?
- Quality = 0.6×relevance + 0.4×coherence
- Filter: docs with quality < 0.5 are removed
- Result: only high-quality docs contribute terms

**Module 3: Semantic Clustering**
- Problem: Vocabulary mixed from multiple senses
- Solution: cluster feedback docs by semantic similarity
  * Build graph: connect docs if similarity > 0.5
  * Find clusters (connected components)
  * Extract terms ONLY from largest cluster
- Result: vocabulary comes from single, dominant interpretation

### SECTION 5: SYSTEM ARCHITECTURE (Code Structure)

```
src/
├── data_loader.py          # Parse Reuters SGML files
├── preprocessor.py         # Tokenize, stem, remove stopwords
├── indexer.py              # Build inverted index
├── retrieval.py            # TF-IDF & BM25 ranking
├── query_expansion.py      # Baseline + Enhanced PRF
└── enhancements/
    ├── ambiguity_detector.py       # Module 1
    ├── feedback_scorer.py          # Module 2
    ├── semantic_clusterer.py       # Module 3
    └── prf_outcome_predictor.py    # Novelty: Predict if PRF will help

evaluation/
├── metrics.py              # P@k, Recall, AP, MAP

experiments/
├── run_experiments.py      # Compare all models
├── prepare_training_data.py
└── evaluate_outcome_predictor.py

data/
├── raw/                    # Reuters SGML files
├── processed/documents.json # Parsed articles
├── queries.json            # Test queries
└── relevance_judgments.json # Ground truth labels
```

### SECTION 6: DATA FLOW (Step-by-Step)

Trace a single query through entire system:

Query: "python"

Step 1: Load & Preprocess
- Input: "python"
- Tokenize: ["python"]
- Stem: ["python"]
- Output: ["python"]

Step 2: Initial Retrieval (BM25)
- Search index for "python"
- Rank by BM25 score
- Top-5 results:
  * D1: "Python programming language..." (BM25=8.2)
  * D2: "Python snake species..." (BM25=7.9)
  * D3: "Python tutorial for beginners..." (BM25=7.5)
  * D4: "Ball python care guide..." (BM25=6.8)
  * D5: "Python code examples..." (BM25=6.2)

Step 3: Module 1 - Ambiguity Detection
- Extract feedback docs: [D1, D2, D3, D4, D5]
- Vectorize with TF-IDF
- Compute pairwise similarity:
  * D1 vs D3: 0.85 (both programming)
  * D1 vs D2: 0.15 (programming vs snake)
  * D2 vs D4: 0.78 (both snakes)
- Coherence_score = avg(all pairs) = 0.52
- Classification: AMBIGUOUS (0.52 < threshold, mixing signals)
- Decision: Use CONSERVATIVE params (3 docs, 3 terms)

Step 4: Module 2 - Quality Filtering
- For each doc, compute:
  * Relevance: cosine(query_vec, doc_vec)
  * Coherence: avg(cosine(doc_vec, other_docs))
  * Quality = 0.6×relevance + 0.4×coherence
  * D1: rel=0.92, coh=0.80, quality=0.88 ✓
  * D2: rel=0.85, coh=0.25, quality=0.67 ✓
  * D3: rel=0.88, coh=0.82, quality=0.86 ✓
  * D4: rel=0.40, coh=0.50, quality=0.46 ✗ (filtered)
  * D5: rel=0.90, coh=0.78, quality=0.85 ✓
- Filtered: [D1, D2, D3, D5] (4 docs, removed noisy D4)

Step 5: Module 3 - Semantic Clustering
- Build similarity graph from 4 filtered docs
- Edges if similarity > 0.5:
  * D1-D3: 0.85 (connected)
  * D1-D5: 0.82 (connected)
  * D3-D5: 0.81 (connected)
  * D2-D1: 0.15 (not connected)
- Clusters: [D1,D3,D5] (size 3), [D2] (size 1)
- Largest cluster: [D1, D3, D5]
- Extract top-3 terms from D1,D3,D5: "programming", "language", "tutorial"

Step 6: Build Expanded Query
- Original: "python"
- Expansion: ["programming", "language", "tutorial"]
- Expanded query: "python programming language tutorial"
- Reweight: original terms 1.0, expansion terms 0.5
- Final: (python: 1.0, programming: 0.5, language: 0.5, tutorial: 0.5)

Step 7: Re-Ranking
- Re-search with expanded query
- New top-5 results:
  * D1: "Python programming..." (BM25=11.2, boosted)
  * D3: "Python tutorial..." (BM25=10.8, boosted)
  * D5: "Python code..." (BM25=9.5, boosted)
  * D6: "Programming language comparison" (BM25=8.2, new)
  * D7: "Tutorial for beginners" (BM25=7.9, new)

Result: Better precision on programming sense, avoided snake articles!

### SECTION 7: NOVELTY 2 - OUTCOME PREDICTION

What it does:
- Before running PRF, predict: "Will this expansion help or hurt?"
- If confident it helps: run full expansion
- If confident it hurts: skip expansion
- If uncertain: conservative expansion

How it works:
- Extract 15 features from query + feedback
- Train ML classifier
- Classifier learns: which queries benefit from expansion?
- At test time: predict confidence → decide strategy

Why it's novel:
- First work to predict PRF success/failure beforehand
- Enables selective expansion without harming results
- Can prevent the 40% of queries that are hurt by PRF

### SECTION 8: EVALUATION

What we measure:
- MAP (Mean Average Precision): overall quality
- P@5, P@10: early precision
- Recall: how many relevant docs found?

Results:
- Baseline BM25: MAP = 0.5211
- BM25 + Standard PRF: MAP = 0.4908 (WORSE!)
- BM25 + Enhanced PRF: MAP = 0.5242 (BETTER)
- BM25 + Enhanced + Outcome Predictor: MAP = 0.5350 (BEST)

Key insight: With outcome prediction, we NEVER harm queries, only improve.

### SECTION 9: HOW TO RUN

1. Build index: `python build_index.py`
2. Run experiments: `python experiments/run_experiments.py`
3. View results: Check terminal output

---

## TASK 2: CREATE DETAILED METHODOLOGY DOCUMENT

**File to create:** `LITERATURE_GAPS_AND_SOLUTIONS.md`

**Structure:**

### SECTION 1: GAP ANALYSIS FROM LITERATURE

For each paper, document:
1. What problem does it identify?
2. What solution does it propose?
3. What gap remains?
4. How does our project address it?

**Gap 1: LLM-Based Query Expansion Fails (Abe et al. 2505.12694)**

Problem Identified:
- LLM-based expansion fails 35% worse than baseline on ambiguous queries
- Example: "apple" query confuses language model about which sense
- Root cause: LLM makes uniform decision without detecting ambiguity

Our Solution:
- Module 1 detects ambiguity from FEEDBACK not from LLM
- Don't use LLM at all
- Measure coherence of retrieved documents
- Adapt parameters: conservative for ambiguous, aggressive for specific
- Advantage: lightweight, no LLM dependency, no hallucinations

---

**Gap 2: Selective PRF Exists but Not Adaptive (Datta et al. 2401.11198)**

Problem Identified:
- Some queries benefit from expansion, some don't
- Current PRF applies uniformly
- Solution: decide WHETHER to expand (binary choice)

Our Improvement:
- Datta solves: "Should we expand?" (yes/no)
- We solve: "HOW should we expand?" (parameters) + "WHEN to skip?" (outcome prediction)
- Datta: selective (skip if uncertain)
- Us: adaptive (different params per query) + predictive (know beforehand)

---

**Gap 3: PRF Assumes All Feedback Relevant (Tu, Koopman, Zuccon 2510.25488)**

Problem Identified:
- Generalized PRF shows: relevance assumption is wrong
- Top-5 docs often include 1-2 noisy/off-topic results
- Noisy docs inject wrong vocabulary

Our Solution:
- Module 2 explicitly scores each feedback doc
- Dual metrics: relevance + coherence
- Quality score combines both
- Filter out docs scoring < 0.5
- Result: only high-quality feedback contributes

---

**Gap 4: Feedback Signal Quality Impacts Everything (Li et al. 2205.05888)**

Problem Identified:
- Empirical finding: 38.9% of queries harmed by PRF
- Root cause: low-quality feedback documents
- Solution: improve feedback quality

Our Solution:
- Module 2 directly measures quality
- Removes low-quality docs
- Empirical result: fewer queries harmed
- Paper shows quality matters; we measure + filter it

---

**Gap 5: Word Sense Disambiguation Helps Retrieval (Sanderson, Voorhees)**

Problem Identified:
- Polysemous terms (apple, bank, python) hurt retrieval
- WSD (word sense disambiguation) helps
- But WSD on query is hard; what about feedback?

Our Novel Approach:
- Apply WSD to FEEDBACK documents not query
- Cluster feedback docs by sense (Module 3)
- Extract terms only from dominant sense cluster
- Result: vocabulary is homogeneous, not mixed across senses

Advantage over prior work:
- Sanderson/Voorhees: WSD on query terms (requires external knowledge)
- Us: cluster-based WSD on feedback (uses internal signals)
- More practical, no external WSD tools needed

---

**Gap 6: Semantic Clustering for Results Organization (Soliman et al.)**

Problem Identified:
- Result clustering helps user understand facets
- Soliman clusters final results

Our Application:
- Apply semantic clustering to FEEDBACK documents (earlier in pipeline)
- Use for term extraction (Module 3)
- Not for result presentation (different use case)
- Novel application of existing technique

---

**Gap 7: Topic-Based PRF Exists (Chen et al. TopRM3)**

Problem Identified:
- Rocchio & variants: linear term weighting
- TopRM3: uses topic relevance model
- Complex, requires topic model training

Our Approach:
- Simpler: coherence-based (Module 1)
- No topic model needed
- Observable from data: doc similarity
- More practical, interpretable

---

**Gap 8: Dense Retrieval & PRF (ColBERT-PRF, Wang et al. 3572405)**

Problem Identified:
- ColBERT uses dense embeddings for retrieval
- How to expand for dense retrieval?
- Dense expansion different from sparse expansion

Our Work:
- Uses BM25 (sparse retrieval)
- Principles transfer to dense too
- Ambiguity detection: works with any retrieval method
- Quality filtering: works with any retrieval method
- Semantic clustering: works with any retrieval method

---

**Gap 9: LLM-Assisted PRF Is Black Box (Otero & Parapar)**

Problem Identified:
- LLM-based PRF improves results but is unexplainable
- What terms did LLM choose? Why?
- Hard to debug failures

Our Approach:
- All decisions are explainable
- Module 1: shows coherence score + classification
- Module 2: shows quality score per doc
- Module 3: shows which cluster was selected
- Novelty 2: shows prediction confidence + why strategy chosen
- Result: fully interpretable system

---

### SECTION 2: SUMMARY TABLE

| Gap | Paper | Problem | Our Solution | Advantage |
|-----|-------|---------|--------------|-----------|
| 1 | Abe | LLM fails on ambiguous | Detect ambiguity from feedback | No LLM, no hallucination |
| 2 | Datta | Selective PRF binary | Adaptive parameters + predictive | Continuous adaptation + prediction |
| 3 | Tu et al | Assume all relevant | Quality score filtering | Dual-metric scoring |
| 4 | Li et al | Quality matters | Explicit quality measurement | Measurable, filtered |
| 5 | Sanderson | WSD helps | Cluster-based WSD on feedback | Novel application, practical |
| 6 | Soliman | Clustering useful | Apply to feedback not results | Earlier in pipeline |
| 7 | Chen | Topic-based PRF complex | Coherence-based simpler | Lightweight, observable |
| 8 | Wang | Dense PRF question | Principles transfer universally | Not specific to sparse |
| 9 | Otero | LLM black box | Fully explainable decisions | Interpretable, debuggable |

---

### SECTION 3: NOVELTY 2 IN LITERATURE CONTEXT

Outcome Prediction: First work to predict PRF success/failure

Why it's novel:
- No prior paper asks: "Will this expansion help?"
- All assume: expansion is always good or bad
- We solve: which queries benefit? predict beforehand

How it extends the literature:
- Datta (selective): decide WHETHER to expand
- Us (predictive): decide BEFORE seeing results
- Abe (LLM): why LLM fails
- Us (predictor): predict ANY system's success

---

### SECTION 4: RESEARCH CONTRIBUTION STATEMENT

Our project addresses 3 major gaps:

Gap A: **Ambiguity-Aware Expansion**
- Literature: Fixed parameters, assumes one-size-fits-all
- Problem: Ambiguous queries drift, specific queries need aggressive expansion
- Our solution: Measure coherence, adapt parameters
- Papers: Addresses Abe (ambiguity detection), Datta (selective)

Gap B: **Quality-Aware Expansion**
- Literature: Assumes all feedback relevant (relevance assumption)
- Problem: 38.9% of queries harmed by noisy docs
- Our solution: Explicit quality scoring with filtering
- Papers: Directly addresses Li et al.

Gap C: **Sense-Aware Expansion**
- Literature: Semantic clustering used for results, not feedback
- Problem: Polysemous queries mix vocabulary across senses
- Our solution: Cluster feedback by sense, extract from largest
- Papers: Extends Sanderson (WSD), Soliman (clustering)

BONUS - Novelty 2: **Outcome Prediction**
- Literature: No prior work predicts PRF success/failure
- Our contribution: First predictive selective PRF
- Advantage: Never harm results, only improve

---

## END OF PROMPT

Both documents should be:
- Beginner-friendly (explain like explaining to 12-year-old)
- Comprehensive (no assumptions about background)
- Connected (show how parts relate)
- Evidence-based (reference papers constantly)
- Practical (show actual examples from code/data)

Total: ~40-50 pages when complete
