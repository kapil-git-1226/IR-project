# Ambiguity-Aware News Retrieval using Query Expansion and Pseudo-Relevance Feedback

**Implementation Blueprint for Semester Project**

---

## 1. Project Overview

### Brief Description
This project implements an Information Retrieval system that addresses query ambiguity in news article search using Query Expansion and Pseudo-Relevance Feedback (PRF). The system retrieves relevant documents from the Reuters-21578 news corpus using TF-IDF and BM25 ranking models, then improves retrieval quality by automatically expanding ambiguous queries with contextually relevant terms.

### Problem Being Solved
Traditional IR systems struggle with ambiguous queries where a single term has multiple meanings:
- **"python"** → programming language or snake species?
- **"jaguar"** → animal, car brand, or sports team?
- **"apple"** → fruit or technology company?
- **"turkey"** → country or bird?

Users searching for news articles often use short, ambiguous queries. Without context, retrieval systems return mixed or irrelevant results.

### Why Query Ambiguity Matters
- **User Intent Mismatch**: 30-40% of web queries are ambiguous
- **Poor User Experience**: Users must sift through irrelevant results
- **Wasted Resources**: Retrieving incorrect documents wastes computational resources
- **Domain-Specific Impact**: In news retrieval, ambiguity is critical since terms often have temporal context (e.g., "corona" pre/post-2020)

### Expected Outcomes
- Demonstrate measurable improvement in retrieval precision using PRF
- Compare TF-IDF vs BM25 effectiveness on ambiguous queries
- Show that query expansion reduces ambiguity impact
- Provide a working prototype with evaluation metrics

---

## 2. System Architecture

### High-Level Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Reuters   │────▶│ Preprocessing│────▶│   Indexing  │────▶│   Storage    │
│  SGML Data  │     │  (Tokenize,  │     │  (Inverted  │     │  (Pickle/    │
│             │     │   Stem, etc.)│     │   Index)    │     │   JSON)      │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Results   │◀────│  Re-Ranking  │◀────│   Scoring   │◀────│  User Query  │
│  (Ranked    │     │              │     │  (TF-IDF/   │     │              │
│   Docs)     │     │              │     │   BM25)     │     │              │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
      │                                          │
      │                                          │
      ▼                                          │
┌─────────────┐     ┌──────────────┐            │
│   Query     │────▶│  Extract Top │            │
│  Expansion  │     │  Terms from  │────────────┘
│  (PRF)      │     │  Top-K Docs  │
└─────────────┘     └──────────────┘
      │
      ▼
┌─────────────┐     ┌──────────────┐
│  Expanded   │────▶│  Re-Execute  │
│   Query     │     │   Retrieval  │
└─────────────┘     └──────────────┘
```

### Component Breakdown

1. **Data Loading & Parsing**
   - Parse Reuters-21578 SGML files
   - Extract title, body, topics
   - Store as structured JSON

2. **Preprocessing**
   - Tokenization: Split text into terms
   - Normalization: Lowercase, remove punctuation
   - Stopword Removal: Filter common words (the, is, and)
   - Stemming: Reduce terms to root form (running → run)

3. **Indexing**
   - Build inverted index: `term → list of (doc_id, positions, term_frequency)`
   - Store document lengths for normalization
   - Calculate document frequency (DF) for each term

4. **Retrieval (TF-IDF & BM25)**
   - **TF-IDF**: Term Frequency × Inverse Document Frequency
   - **BM25**: Probabilistic ranking function with saturation
   - Both produce relevance scores for query-document pairs

5. **Query Expansion (PRF)**
   - Retrieve top-K documents (K=5-10)
   - Extract most frequent/important terms from these documents
   - Add top-M terms (M=5-10) to original query
   - Re-execute retrieval with expanded query

6. **Evaluation**
   - Compare baseline vs expanded queries
   - Measure Precision@K, Recall, MAP
   - Analyze effectiveness on ambiguous vs specific queries

### Where Models Fit
- **TF-IDF**: Used in both initial retrieval and term selection for PRF
- **BM25**: Alternative ranking model, generally more effective than TF-IDF
- **PRF**: Post-processing step that improves both TF-IDF and BM25 results

---

## 3. Tech Stack

### Programming Language
**Python 3.8+**
- Reason: Rich ecosystem for NLP and IR, rapid prototyping
- Easy integration of libraries
- Good performance for academic projects

### Core Libraries

| Library | Purpose | Installation |
|---------|---------|--------------|
| **NLTK** | Tokenization, stopwords, stemming | `pip install nltk` |
| **NumPy** | Numerical computations, vector operations | `pip install numpy` |
| **SciPy** | Sparse matrix operations (optional) | `pip install scipy` |
| **BeautifulSoup4** | SGML/XML parsing | `pip install beautifulsoup4 lxml` |
| **scikit-learn** | Evaluation metrics (optional) | `pip install scikit-learn` |

### Utilities
- **json**: Store processed documents and queries
- **pickle**: Serialize inverted index
- **argparse**: CLI interface
- **collections.defaultdict**: Efficient dictionary for indexing
- **re**: Regular expressions for tokenization
- **math**: Log functions for scoring

### Why These Choices?
- **NLTK over spaCy**: NLTK is lighter and sufficient for basic NLP tasks
- **In-memory indexing**: Simpler than disk-based systems (Whoosh, Elasticsearch) for small datasets
- **No deep learning**: Keeps project scope manageable
- **Standard libraries**: Minimize dependencies

### Development Environment
```bash
# Create virtual environment
python -m venv venvcd
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install nltk numpy beautifulsoup4 lxml scikit-learn

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 4. Data Handling

### Reuters-21578 Dataset

**Overview**:
- 21,578 news articles from Reuters newswire (1987)
- 22 SGML files (reut2-000.sgm to reut2-021.sgm)
- Articles cover finance, economics, politics
- Download: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection)

### SGML Structure
```xml
<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="5544" NEWID="1">
<DATE>26-FEB-1987 15:01:01.79</DATE>
<TOPICS><D>cocoa</D></TOPICS>
<PLACES><D>el-salvador</D><D>usa</D></PLACES>
<TITLE>BAHIA COCOA REVIEW</TITLE>
<BODY>The Bahia cocoa trade is now...</BODY>
</REUTERS>
```

### Fields to Extract
- **NEWID**: Unique document identifier
- **TITLE**: Article headline (may be empty)
- **BODY**: Article content (main text)
- **TOPICS**: Category tags (optional, for filtering)
- **DATE**: Publication date (optional)

### SGML Parser Implementation

```python
from bs4 import BeautifulSoup
import re
import json

def parse_reuters_sgml(file_path):
    """Parse a single Reuters SGML file."""
    with open(file_path, 'r', encoding='latin-1') as f:
        content = f.read()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(content, 'lxml')
    documents = []

    for article in soup.find_all('reuters'):
        doc_id = article.get('newid')

        # Extract fields
        title = article.find('title')
        title_text = title.get_text() if title else ""

        body = article.find('body')
        body_text = body.get_text() if body else ""

        topics = [d.get_text() for d in article.find_all('d')] if article.find('topics') else []

        # Skip empty documents
        if not title_text and not body_text:
            continue

        documents.append({
            'doc_id': doc_id,
            'title': title_text.strip(),
            'body': body_text.strip(),
            'topics': topics,
            'text': f"{title_text} {body_text}".strip()  # Combined text
        })

    return documents

def parse_all_reuters(data_dir):
    """Parse all Reuters SGML files."""
    all_docs = []
    for i in range(22):  # 000 to 021
        file_name = f"reut2-{i:03d}.sgm"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            docs = parse_reuters_sgml(file_path)
            all_docs.extend(docs)
            print(f"Parsed {file_name}: {len(docs)} documents")

    return all_docs
```

### Data Cleaning Steps

```python
def clean_text(text):
    """Clean and normalize text."""
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)

    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def save_processed_documents(documents, output_path):
    """Save processed documents to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
```

### Data Loading Workflow
```python
# Step 1: Parse SGML files
documents = parse_all_reuters('data/raw/')

# Step 2: Clean text
for doc in documents:
    doc['text'] = clean_text(doc['text'])

# Step 3: Save to JSON
save_processed_documents(documents, 'data/processed/documents.json')

print(f"Total documents: {len(documents)}")
```

---

## 5. Preprocessing Pipeline

### Overview
Preprocessing transforms raw text into normalized tokens suitable for indexing and retrieval.

### Step 1: Tokenization

```python
import re

def tokenize(text):
    """Split text into tokens using regex."""
    # Convert to lowercase
    text = text.lower()

    # Extract words (alphanumeric sequences)
    tokens = re.findall(r'\b[a-z0-9]+\b', text)

    return tokens

# Example
text = "Python's effectiveness in IR systems!"
tokens = tokenize(text)  # ['python', 's', 'effectiveness', 'in', 'ir', 'systems']
```

### Step 2: Stopword Removal

```python
from nltk.corpus import stopwords

# Load English stopwords
STOP_WORDS = set(stopwords.words('english'))

def remove_stopwords(tokens):
    """Filter out common stopwords."""
    return [token for token in tokens if token not in STOP_WORDS]

# Example
filtered = remove_stopwords(['python', 's', 'effectiveness', 'in', 'ir', 'systems'])
# Result: ['python', 's', 'effectiveness', 'ir', 'systems']
```

### Step 3: Stemming

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens):
    """Apply Porter Stemmer to reduce words to root form."""
    return [stemmer.stem(token) for token in tokens]

# Example
stemmed = stem_tokens(['running', 'runs', 'runner', 'easily', 'fairly'])
# Result: ['run', 'run', 'runner', 'easili', 'fairli']
```

**Note**: Stemming may produce non-words (e.g., "easili"), but this is acceptable for IR.

**Alternative - Lemmatization** (more accurate but slower):
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
```

### Complete Preprocessing Function

```python
class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        """Complete preprocessing pipeline."""
        # Tokenize
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())

        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # Stem
        tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens

# Usage
preprocessor = TextPreprocessor()
tokens = preprocessor.preprocess("Python's effectiveness in IR systems!")
# Result: ['python', 'effect', 'ir', 'system']
```

### Vocabulary Building

```python
class Vocabulary:
    def __init__(self):
        self.term_to_id = {}
        self.id_to_term = {}
        self.next_id = 0

    def add_term(self, term):
        """Add term to vocabulary if not exists."""
        if term not in self.term_to_id:
            self.term_to_id[term] = self.next_id
            self.id_to_term[self.next_id] = term
            self.next_id += 1
        return self.term_to_id[term]

    def get_id(self, term):
        """Get term ID."""
        return self.term_to_id.get(term, -1)

    def get_term(self, term_id):
        """Get term from ID."""
        return self.id_to_term.get(term_id, None)

    def __len__(self):
        return len(self.term_to_id)
```

---

## 6. Indexing

### Inverted Index Structure

An inverted index maps each term to a list of documents containing that term:

```
{
    'python': [(doc1, 3), (doc5, 1), (doc12, 2)],  # (doc_id, term_frequency)
    'retriev': [(doc1, 2), (doc3, 5), (doc8, 1)],
    ...
}
```

### Data Structures

```python
from collections import defaultdict, Counter
import pickle
import math

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)  # term -> [(doc_id, tf), ...]
        self.doc_lengths = {}  # doc_id -> document length
        self.doc_count = 0
        self.avg_doc_length = 0
        self.vocabulary = Vocabulary()

    def build_index(self, documents, preprocessor):
        """Build inverted index from documents."""
        self.doc_count = len(documents)
        total_length = 0

        for doc in documents:
            doc_id = doc['doc_id']
            text = doc['text']

            # Preprocess text
            tokens = preprocessor.preprocess(text)

            # Count term frequencies
            term_freq = Counter(tokens)

            # Store document length
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Add to inverted index
            for term, tf in term_freq.items():
                term_id = self.vocabulary.add_term(term)
                self.index[term].append((doc_id, tf))

        # Calculate average document length
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0

        print(f"Index built: {len(self.index)} unique terms, {self.doc_count} documents")

    def get_postings(self, term):
        """Get postings list for a term."""
        return self.index.get(term, [])

    def get_document_frequency(self, term):
        """Get number of documents containing term."""
        return len(self.index.get(term, []))

    def save(self, file_path):
        """Serialize index to disk."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """Load index from disk."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
```

### Storage Approach

**In-Memory with Serialization**:
- Store index in RAM during indexing and retrieval
- Serialize to disk using pickle for persistence
- Fast for datasets up to 100K documents
- Suitable for Reuters-21578 (21K documents)

**Alternative for Larger Datasets**:
- Use disk-based systems (Whoosh, Elasticsearch)
- Store index in SQLite for term → postings mapping

### Building the Index

```python
# Load documents
with open('data/processed/documents.json', 'r') as f:
    documents = json.load(f)

# Create preprocessor and index
preprocessor = TextPreprocessor()
index = InvertedIndex()

# Build index
index.build_index(documents, preprocessor)

# Save to disk
index.save('data/index/inverted_index.pkl')

# Statistics
print(f"Vocabulary size: {len(index.vocabulary)}")
print(f"Average document length: {index.avg_doc_length:.2f}")
```

---

## 7. Retrieval Models

### TF-IDF Implementation

**Formula**:
```
score(q, d) = Σ (tf(t, d) × idf(t))
              t∈q

where:
- tf(t, d) = frequency of term t in document d
- idf(t) = log(N / df(t))
- N = total number of documents
- df(t) = number of documents containing term t
```

**Implementation**:

```python
class TFIDFRetrieval:
    def __init__(self, index):
        self.index = index

    def calculate_idf(self, term):
        """Calculate IDF for a term."""
        df = self.index.get_document_frequency(term)
        if df == 0:
            return 0
        return math.log(self.index.doc_count / df)

    def calculate_tf(self, term_freq):
        """Calculate TF (using raw frequency)."""
        return term_freq

    def score_document(self, query_terms, doc_id, postings_dict):
        """Calculate TF-IDF score for a document."""
        score = 0.0

        for term in query_terms:
            if term not in postings_dict:
                continue

            # Get term frequency in document
            tf = postings_dict[term].get(doc_id, 0)
            if tf == 0:
                continue

            # Calculate IDF
            idf = self.calculate_idf(term)

            # Add to score
            score += tf * idf

        return score

    def search(self, query, preprocessor, top_k=10):
        """Search for documents matching query."""
        # Preprocess query
        query_terms = preprocessor.preprocess(query)

        # Get postings for each query term
        postings_dict = {}
        candidate_docs = set()

        for term in query_terms:
            postings = self.index.get_postings(term)
            postings_dict[term] = {doc_id: tf for doc_id, tf in postings}
            candidate_docs.update(postings_dict[term].keys())

        # Score each candidate document
        scores = []
        for doc_id in candidate_docs:
            score = self.score_document(query_terms, doc_id, postings_dict)
            scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
```

### BM25 Implementation

**Formula**:
```
score(q, d) = Σ idf(t) × (tf(t, d) × (k1 + 1)) / (tf(t, d) + k1 × (1 - b + b × (|d| / avgdl)))
              t∈q

where:
- k1 = term frequency saturation parameter (typically 1.2-2.0)
- b = length normalization parameter (typically 0.75)
- |d| = document length
- avgdl = average document length
```

**Implementation**:

```python
class BM25Retrieval:
    def __init__(self, index, k1=1.5, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def calculate_idf(self, term):
        """Calculate IDF for BM25."""
        df = self.index.get_document_frequency(term)
        if df == 0:
            return 0
        # BM25 IDF variant
        return math.log((self.index.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def score_document(self, query_terms, doc_id, postings_dict):
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        avgdl = self.index.avg_doc_length

        for term in query_terms:
            if term not in postings_dict:
                continue

            # Get term frequency in document
            tf = postings_dict[term].get(doc_id, 0)
            if tf == 0:
                continue

            # Calculate IDF
            idf = self.calculate_idf(term)

            # Calculate BM25 component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avgdl))

            score += idf * (numerator / denominator)

        return score

    def search(self, query, preprocessor, top_k=10):
        """Search for documents matching query."""
        # Preprocess query
        query_terms = preprocessor.preprocess(query)

        # Get postings for each query term
        postings_dict = {}
        candidate_docs = set()

        for term in query_terms:
            postings = self.index.get_postings(term)
            postings_dict[term] = {doc_id: tf for doc_id, tf in postings}
            candidate_docs.update(postings_dict[term].keys())

        # Score each candidate document
        scores = []
        for doc_id in candidate_docs:
            score = self.score_document(query_terms, doc_id, postings_dict)
            scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
```

### Query Processing Pipeline

```python
def process_query(query, index, preprocessor, model='bm25', top_k=10):
    """
    Process a query and return results.

    Args:
        query: Raw query string
        index: InvertedIndex instance
        preprocessor: TextPreprocessor instance
        model: 'tfidf' or 'bm25'
        top_k: Number of results to return

    Returns:
        List of (doc_id, score) tuples
    """
    if model == 'tfidf':
        retrieval = TFIDFRetrieval(index)
    elif model == 'bm25':
        retrieval = BM25Retrieval(index)
    else:
        raise ValueError(f"Unknown model: {model}")

    results = retrieval.search(query, preprocessor, top_k)
    return results
```

---

## 8. Query Expansion (Pseudo-Relevance Feedback)

### Overview
PRF assumes the top-K retrieved documents are relevant and uses them to refine the query by adding related terms.

### Step-by-Step PRF Workflow

```python
class PseudoRelevanceFeedback:
    def __init__(self, index, preprocessor):
        self.index = index
        self.preprocessor = preprocessor

    def extract_expansion_terms(self, doc_ids, top_m=10):
        """
        Extract top-M expansion terms from a set of documents.

        Args:
            doc_ids: List of document IDs
            top_m: Number of expansion terms to extract

        Returns:
            List of expansion terms
        """
        term_scores = defaultdict(float)

        # For each document, calculate term importance
        for doc_id in doc_ids:
            # Get all terms in document (from index postings)
            for term, postings in self.index.index.items():
                for pid, tf in postings:
                    if pid == doc_id:
                        # Score term by TF-IDF
                        idf = math.log(self.index.doc_count / len(postings))
                        term_scores[term] += tf * idf
                        break

        # Sort terms by score
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-M terms (excluding very common terms)
        expansion_terms = [term for term, score in sorted_terms[:top_m * 2]]
        return expansion_terms[:top_m]

    def expand_query(self, original_query, expansion_terms, alpha=1.0, beta=0.5):
        """
        Create expanded query by combining original and expansion terms.

        Args:
            original_query: List of original query terms
            expansion_terms: List of expansion terms
            alpha: Weight for original query
            beta: Weight for expansion terms

        Returns:
            Expanded query as a string
        """
        # Weight original query terms
        expanded = original_query * int(alpha)

        # Add expansion terms with weight
        expanded.extend(expansion_terms[:int(len(expansion_terms) * beta)])

        return ' '.join(expanded)

    def search_with_prf(self, query, retrieval_model, top_k=10, feedback_docs=5, expansion_terms=10):
        """
        Perform retrieval with PRF.

        Args:
            query: Original query string
            retrieval_model: TFIDFRetrieval or BM25Retrieval instance
            top_k: Number of final results
            feedback_docs: Number of top documents to use for feedback
            expansion_terms: Number of terms to add to query

        Returns:
            Tuple of (final_results, expanded_query_string)
        """
        # Step 1: Initial retrieval
        initial_results = retrieval_model.search(query, self.preprocessor, top_k=feedback_docs)

        if not initial_results:
            return [], query

        # Step 2: Extract document IDs from top results
        top_doc_ids = [doc_id for doc_id, score in initial_results]

        # Step 3: Extract expansion terms
        expansion_terms_list = self.extract_expansion_terms(top_doc_ids, top_m=expansion_terms)

        # Step 4: Form expanded query
        original_terms = self.preprocessor.preprocess(query)
        expanded_query = self.expand_query(original_terms, expansion_terms_list)

        # Step 5: Re-execute retrieval with expanded query
        final_results = retrieval_model.search(expanded_query, self.preprocessor, top_k)

        return final_results, expanded_query
```

### Rocchio Algorithm (Optional)

The Rocchio algorithm provides a more principled approach to query expansion:

```python
class RocchioPRF:
    def __init__(self, index, preprocessor, alpha=1.0, beta=0.75, gamma=0.0):
        """
        Rocchio algorithm for query expansion.

        Args:
            alpha: Weight for original query
            beta: Weight for relevant documents
            gamma: Weight for non-relevant documents (typically 0 for PRF)
        """
        self.index = index
        self.preprocessor = preprocessor
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_document_vector(self, doc_id):
        """Get TF-IDF vector for a document."""
        vector = {}
        for term, postings in self.index.index.items():
            for pid, tf in postings:
                if pid == doc_id:
                    idf = math.log(self.index.doc_count / len(postings))
                    vector[term] = tf * idf
                    break
        return vector

    def expand_query_rocchio(self, original_query, relevant_doc_ids, top_m=10):
        """
        Expand query using Rocchio formula.

        Q_new = α * Q_original + (β / |R|) * Σ D_relevant
        """
        # Get original query vector (binary: 1 if term in query, 0 otherwise)
        query_terms = self.preprocessor.preprocess(original_query)
        query_vector = {term: 1.0 for term in query_terms}

        # Calculate centroid of relevant documents
        relevant_centroid = defaultdict(float)
        for doc_id in relevant_doc_ids:
            doc_vector = self.get_document_vector(doc_id)
            for term, weight in doc_vector.items():
                relevant_centroid[term] += weight

        # Average
        if relevant_doc_ids:
            for term in relevant_centroid:
                relevant_centroid[term] /= len(relevant_doc_ids)

        # Compute new query vector: α * Q + β * R
        new_query_vector = {}

        # Add weighted original query
        for term, weight in query_vector.items():
            new_query_vector[term] = self.alpha * weight

        # Add weighted relevant documents
        for term, weight in relevant_centroid.items():
            new_query_vector[term] = new_query_vector.get(term, 0) + self.beta * weight

        # Select top-M terms
        sorted_terms = sorted(new_query_vector.items(), key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, weight in sorted_terms[:top_m]]

        return ' '.join(expansion_terms)
```

### Usage Example

```python
# Load index
index = InvertedIndex.load('data/index/inverted_index.pkl')
preprocessor = TextPreprocessor()

# Initialize retrieval and PRF
retrieval = BM25Retrieval(index)
prf = PseudoRelevanceFeedback(index, preprocessor)

# Search with PRF
query = "python programming"
results, expanded_query = prf.search_with_prf(
    query,
    retrieval,
    top_k=10,
    feedback_docs=5,
    expansion_terms=8
)

print(f"Original query: {query}")
print(f"Expanded query: {expanded_query}")
print("\nResults:")
for doc_id, score in results:
    print(f"  {doc_id}: {score:.4f}")
```

---

## 9. Evaluation Strategy

### Test Query Design

Create a diverse set of test queries to evaluate the system:

**Query Categories**:
1. **Ambiguous Queries** (5 queries):
   - "python" (programming vs snake)
   - "turkey" (country vs bird)
   - "apple" (fruit vs company)
   - "jaguar" (animal vs car)
   - "java" (island vs programming)

2. **Specific Queries** (10 queries):
   - "stock market crash"
   - "oil prices middle east"
   - "space shuttle launch"
   - "economic recession impact"
   - "international trade agreement"
   - "presidential election results"
   - "natural disaster relief"
   - "corporate merger announcement"
   - "interest rate policy"
   - "unemployment statistics"

**queries.json**:
```json
[
  {
    "query_id": "Q001",
    "text": "python",
    "type": "ambiguous",
    "description": "Programming language or snake species"
  },
  {
    "query_id": "Q002",
    "text": "stock market crash",
    "type": "specific",
    "description": "Financial market decline"
  }
]
```

### Relevance Labeling Strategy

**Manual Annotation Process**:
1. Execute each query without expansion
2. Retrieve top-20 documents
3. Read each document and label as relevant (1) or non-relevant (0)
4. Store judgments in JSON

**Relevance Criteria**:
- Document must address the query intent
- For ambiguous queries, decide on one interpretation and label consistently
- Partial relevance counts as relevant (binary: 0 or 1)

**relevance_judgments.json**:
```json
{
  "Q001": {
    "doc1": 1,
    "doc5": 1,
    "doc12": 0,
    "doc23": 1
  },
  "Q002": {
    "doc3": 1,
    "doc7": 0
  }
}
```

### Evaluation Metrics

#### Precision@K

```python
def precision_at_k(retrieved_docs, relevant_docs, k):
    """
    Calculate Precision at K.

    Args:
        retrieved_docs: List of retrieved doc IDs (in rank order)
        relevant_docs: Set of relevant doc IDs
        k: Cutoff rank

    Returns:
        Precision@K value
    """
    if k == 0:
        return 0.0

    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_docs)

    return relevant_retrieved / k
```

#### Recall

```python
def recall(retrieved_docs, relevant_docs):
    """
    Calculate Recall.

    Args:
        retrieved_docs: List of retrieved doc IDs
        relevant_docs: Set of relevant doc IDs

    Returns:
        Recall value
    """
    if len(relevant_docs) == 0:
        return 0.0

    relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)

    return relevant_retrieved / len(relevant_docs)
```

#### Average Precision (AP)

```python
def average_precision(retrieved_docs, relevant_docs):
    """
    Calculate Average Precision.

    Args:
        retrieved_docs: List of retrieved doc IDs (in rank order)
        relevant_docs: Set of relevant doc IDs

    Returns:
        Average Precision value
    """
    if len(relevant_docs) == 0:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / len(relevant_docs) if relevant_count > 0 else 0.0
```

#### Mean Average Precision (MAP)

```python
def mean_average_precision(query_results, relevance_judgments):
    """
    Calculate Mean Average Precision across all queries.

    Args:
        query_results: Dict {query_id: [retrieved_doc_ids]}
        relevance_judgments: Dict {query_id: {doc_id: relevance}}

    Returns:
        MAP value
    """
    ap_sum = 0.0
    query_count = 0

    for query_id, retrieved_docs in query_results.items():
        if query_id not in relevance_judgments:
            continue

        relevant_docs = {doc_id for doc_id, rel in relevance_judgments[query_id].items() if rel == 1}
        ap = average_precision(retrieved_docs, relevant_docs)
        ap_sum += ap
        query_count += 1

    return ap_sum / query_count if query_count > 0 else 0.0
```

### Evaluation Script

```python
def evaluate_retrieval(queries, retrieval_model, preprocessor, relevance_judgments, use_prf=False):
    """
    Evaluate retrieval system on a set of queries.

    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'p@5': [],
        'p@10': [],
        'recall': [],
        'ap': []
    }

    for query_data in queries:
        query_id = query_data['query_id']
        query_text = query_data['text']

        # Retrieve documents
        if use_prf:
            prf = PseudoRelevanceFeedback(retrieval_model.index, preprocessor)
            retrieved, _ = prf.search_with_prf(query_text, retrieval_model, top_k=20)
        else:
            retrieved = retrieval_model.search(query_text, preprocessor, top_k=20)

        retrieved_docs = [doc_id for doc_id, score in retrieved]

        # Get relevant documents
        if query_id not in relevance_judgments:
            continue

        relevant_docs = {doc_id for doc_id, rel in relevance_judgments[query_id].items() if rel == 1}

        # Calculate metrics
        p5 = precision_at_k(retrieved_docs, relevant_docs, 5)
        p10 = precision_at_k(retrieved_docs, relevant_docs, 10)
        rec = recall(retrieved_docs, relevant_docs)
        ap = average_precision(retrieved_docs, relevant_docs)

        results['p@5'].append(p5)
        results['p@10'].append(p10)
        results['recall'].append(rec)
        results['ap'].append(ap)

    # Calculate averages
    avg_results = {
        'P@5': sum(results['p@5']) / len(results['p@5']) if results['p@5'] else 0,
        'P@10': sum(results['p@10']) / len(results['p@10']) if results['p@10'] else 0,
        'Recall': sum(results['recall']) / len(results['recall']) if results['recall'] else 0,
        'MAP': sum(results['ap']) / len(results['ap']) if results['ap'] else 0
    }

    return avg_results
```

---

## 10. Experiment Design

### Experimental Configurations

Test 4 configurations:
1. **Baseline TF-IDF**: Original queries with TF-IDF
2. **Expanded TF-IDF**: PRF-expanded queries with TF-IDF
3. **Baseline BM25**: Original queries with BM25
4. **Expanded BM25**: PRF-expanded queries with BM25

### Experiment Runner

```python
def run_all_experiments(index, queries, relevance_judgments):
    """Run all experimental configurations."""
    preprocessor = TextPreprocessor()

    # Configuration 1: Baseline TF-IDF
    print("Running Baseline TF-IDF...")
    tfidf_model = TFIDFRetrieval(index)
    baseline_tfidf = evaluate_retrieval(queries, tfidf_model, preprocessor, relevance_judgments, use_prf=False)

    # Configuration 2: Expanded TF-IDF
    print("Running Expanded TF-IDF...")
    expanded_tfidf = evaluate_retrieval(queries, tfidf_model, preprocessor, relevance_judgments, use_prf=True)

    # Configuration 3: Baseline BM25
    print("Running Baseline BM25...")
    bm25_model = BM25Retrieval(index)
    baseline_bm25 = evaluate_retrieval(queries, bm25_model, preprocessor, relevance_judgments, use_prf=False)

    # Configuration 4: Expanded BM25
    print("Running Expanded BM25...")
    expanded_bm25 = evaluate_retrieval(queries, bm25_model, preprocessor, relevance_judgments, use_prf=True)

    # Compile results
    results = {
        'Baseline TF-IDF': baseline_tfidf,
        'Expanded TF-IDF': expanded_tfidf,
        'Baseline BM25': baseline_bm25,
        'Expanded BM25': expanded_bm25
    }

    return results
```

### Results Visualization

```python
import json

def print_results_table(results):
    """Print results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"{'Configuration':<25} {'P@5':>10} {'P@10':>10} {'Recall':>10} {'MAP':>10}")
    print("=" * 70)

    for config, metrics in results.items():
        print(f"{config:<25} {metrics['P@5']:>10.4f} {metrics['P@10']:>10.4f} "
              f"{metrics['Recall']:>10.4f} {metrics['MAP']:>10.4f}")

    print("=" * 70)

def save_results(results, file_path):
    """Save results to JSON."""
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)
```

### Analysis Focus

**Key Comparisons**:
1. **PRF Effectiveness**: Compare Baseline vs Expanded for each model
   - Expected: Expanded queries improve P@5, P@10, and MAP
   - Especially significant for ambiguous queries

2. **Model Comparison**: Compare TF-IDF vs BM25
   - Expected: BM25 outperforms TF-IDF due to length normalization
   - More pronounced on longer documents

3. **Query Type Analysis**: Break down results by ambiguous vs specific queries
   - Expected: PRF shows greater improvement on ambiguous queries

**Statistical Significance** (optional):
```python
from scipy.stats import ttest_rel

def compare_configurations(results1, results2, metric='MAP'):
    """Perform paired t-test on two configurations."""
    values1 = results1[metric]
    values2 = results2[metric]

    t_stat, p_value = ttest_rel(values1, values2)

    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Difference is statistically significant (p < 0.05)")
    else:
        print("Difference is not statistically significant")
```

---

## 11. Project Structure (Code Organization)

### Directory Layout

```
IR-project/
│
├── data/
│   ├── raw/                          # Original Reuters SGML files
│   │   ├── reut2-000.sgm
│   │   ├── reut2-001.sgm
│   │   └── ...
│   │
│   ├── processed/                    # Parsed and cleaned documents
│   │   └── documents.json
│   │
│   └── index/                        # Serialized index
│       └── inverted_index.pkl
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_loader.py               # SGML parsing and data loading
│   │   ├── parse_reuters_sgml()
│   │   ├── parse_all_reuters()
│   │   └── clean_text()
│   │
│   ├── preprocessor.py              # Text preprocessing
│   │   ├── TextPreprocessor class
│   │   └── Vocabulary class
│   │
│   ├── indexer.py                   # Inverted index implementation
│   │   └── InvertedIndex class
│   │
│   ├── retrieval.py                 # Retrieval models
│   │   ├── TFIDFRetrieval class
│   │   └── BM25Retrieval class
│   │
│   ├── query_expansion.py           # Query expansion methods
│   │   ├── PseudoRelevanceFeedback class
│   │   └── RocchioPRF class (optional)
│   │
│   └── main.py                      # CLI interface
│       └── main()
│
├── evaluation/
│   ├── __init__.py
│   │
│   ├── metrics.py                   # Evaluation metrics
│   │   ├── precision_at_k()
│   │   ├── recall()
│   │   ├── average_precision()
│   │   └── mean_average_precision()
│   │
│   ├── queries.json                 # Test queries
│   └── relevance_judgments.json     # Manual relevance labels
│
├── experiments/
│   └── run_experiments.py           # Automated experiment runner
│       ├── run_all_experiments()
│       └── print_results_table()
│
├── utils/
│   └── helpers.py                   # Utility functions
│       ├── load_json()
│       └── save_json()
│
├── results/                         # Experiment outputs
│   ├── experiment_results.json
│   └── logs/
│
├── notebooks/                       # Jupyter notebooks (optional)
│   ├── data_exploration.ipynb
│   └── result_analysis.ipynb
│
├── requirements.txt                 # Python dependencies
├── MASTER.md                        # This file
├── README.md                        # Project summary
└── .gitignore
```

### File Responsibilities

**src/data_loader.py**:
- Parse Reuters SGML files
- Extract title, body, topics
- Clean and normalize text
- Save to JSON format

**src/preprocessor.py**:
- Tokenization
- Stopword removal
- Stemming/lemmatization
- Vocabulary management

**src/indexer.py**:
- Build inverted index
- Store document statistics
- Serialize/deserialize index
- Provide posting list access

**src/retrieval.py**:
- Implement TF-IDF scoring
- Implement BM25 scoring
- Query processing
- Result ranking

**src/query_expansion.py**:
- Implement PRF
- Extract expansion terms
- Form expanded queries
- Optional: Rocchio algorithm

**src/main.py**:
- Command-line interface
- Orchestrate pipeline components
- Handle user interaction

**evaluation/metrics.py**:
- Implement P@K, Recall, AP, MAP
- Evaluate retrieval results
- Generate performance reports

**experiments/run_experiments.py**:
- Run all configurations
- Compare models
- Generate comparison tables
- Save results

---

## 12. Minimal UI / CLI

### Command-Line Interface

```python
# src/main.py
import argparse
import json
from preprocessor import TextPreprocessor
from indexer import InvertedIndex
from retrieval import TFIDFRetrieval, BM25Retrieval
from query_expansion import PseudoRelevanceFeedback

def load_documents(doc_path):
    """Load processed documents."""
    with open(doc_path, 'r') as f:
        return json.load(f)

def build_index_command(args):
    """Build and save inverted index."""
    print("Loading documents...")
    documents = load_documents(args.documents)

    print("Building index...")
    preprocessor = TextPreprocessor()
    index = InvertedIndex()
    index.build_index(documents, preprocessor)

    print(f"Saving index to {args.output}...")
    index.save(args.output)

    print("Done!")
    print(f"Indexed {index.doc_count} documents with {len(index.vocabulary)} unique terms")

def search_command(args):
    """Execute search query."""
    print("Loading index...")
    index = InvertedIndex.load(args.index)

    print("Loading documents for display...")
    documents = load_documents(args.documents)
    doc_dict = {doc['doc_id']: doc for doc in documents}

    preprocessor = TextPreprocessor()

    # Select retrieval model
    if args.model == 'tfidf':
        retrieval = TFIDFRetrieval(index)
    elif args.model == 'bm25':
        retrieval = BM25Retrieval(index)
    else:
        print(f"Unknown model: {args.model}")
        return

    # Execute search
    if args.expand:
        print("Using Query Expansion (PRF)...")
        prf = PseudoRelevanceFeedback(index, preprocessor)
        results, expanded_query = prf.search_with_prf(
            args.query,
            retrieval,
            top_k=args.top_k,
            feedback_docs=args.feedback_docs,
            expansion_terms=args.expansion_terms
        )
        print(f"\nOriginal query: {args.query}")
        print(f"Expanded query: {expanded_query}")
    else:
        results = retrieval.search(args.query, preprocessor, top_k=args.top_k)

    # Display results
    print(f"\n{'='*80}")
    print(f"Top {len(results)} Results:")
    print(f"{'='*80}\n")

    for rank, (doc_id, score) in enumerate(results, 1):
        doc = doc_dict.get(doc_id, {})
        title = doc.get('title', 'No Title')
        body = doc.get('body', '')

        # Create snippet
        snippet = body[:200] + "..." if len(body) > 200 else body

        print(f"{rank}. [Doc {doc_id}] (Score: {score:.4f})")
        print(f"   Title: {title}")
        print(f"   Snippet: {snippet}\n")

def main():
    parser = argparse.ArgumentParser(description='IR System with Query Expansion')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Index command
    index_parser = subparsers.add_parser('index', help='Build inverted index')
    index_parser.add_argument('--documents', default='data/processed/documents.json',
                              help='Path to processed documents')
    index_parser.add_argument('--output', default='data/index/inverted_index.pkl',
                              help='Output path for index')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--index', default='data/index/inverted_index.pkl',
                               help='Path to inverted index')
    search_parser.add_argument('--documents', default='data/processed/documents.json',
                               help='Path to documents')
    search_parser.add_argument('--model', choices=['tfidf', 'bm25'], default='bm25',
                               help='Retrieval model')
    search_parser.add_argument('--top-k', type=int, default=10,
                               help='Number of results to return')
    search_parser.add_argument('--expand', action='store_true',
                               help='Use query expansion (PRF)')
    search_parser.add_argument('--feedback-docs', type=int, default=5,
                               help='Number of documents for PRF')
    search_parser.add_argument('--expansion-terms', type=int, default=10,
                               help='Number of terms to add in expansion')

    args = parser.parse_args()

    if args.command == 'index':
        build_index_command(args)
    elif args.command == 'search':
        search_command(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
```

### Usage Examples

```bash
# Build index
python src/main.py index

# Search with BM25 (default)
python src/main.py search "python programming"

# Search with TF-IDF
python src/main.py search "stock market crash" --model tfidf

# Search with query expansion
python src/main.py search "python" --expand

# Search with custom parameters
python src/main.py search "turkey" --expand --feedback-docs 10 --expansion-terms 8 --top-k 20

# Search with TF-IDF and expansion
python src/main.py search "jaguar" --model tfidf --expand
```

### Output Format

```
Loading index...
Using Query Expansion (PRF)...

Original query: python
Expanded query: python programming language code software development interpreter script

================================================================================
Top 10 Results:
================================================================================

1. [Doc 1523] (Score: 12.3456)
   Title: Python Gains Popularity in Data Science
   Snippet: Python has become the most popular programming language for data science applications, surpassing R and MATLAB in recent surveys...

2. [Doc 3421] (Score: 11.2341)
   Title: New Python 3.9 Release Announced
   Snippet: The Python Software Foundation announced the release of Python 3.9, featuring improved performance and new syntax features...

[...]
```

---

## 13. Timeline / Milestones

### 8-Week Project Timeline

#### Week 1: Data Loading and Preprocessing
**Goals**:
- Set up project structure
- Download Reuters-21578 dataset
- Implement SGML parser
- Implement preprocessing pipeline

**Tasks**:
- [ ] Create directory structure
- [ ] Install dependencies (NLTK, BeautifulSoup, etc.)
- [ ] Write `data_loader.py` with SGML parsing
- [ ] Write `preprocessor.py` with tokenization, stemming
- [ ] Parse all Reuters files and save to JSON
- [ ] Verify: 21K+ documents processed successfully

**Deliverable**: Processed documents in JSON format

---

#### Week 2: Indexing Implementation
**Goals**:
- Implement inverted index
- Build index from processed documents
- Test index performance

**Tasks**:
- [ ] Write `InvertedIndex` class in `indexer.py`
- [ ] Implement index building with term frequencies
- [ ] Store document lengths and collection statistics
- [ ] Serialize index to disk using pickle
- [ ] Test loading and querying index
- [ ] Verify: Index contains ~50K-100K unique terms

**Deliverable**: Serialized inverted index file

---

#### Week 3: TF-IDF Retrieval
**Goals**:
- Implement TF-IDF scoring
- Build query processing pipeline
- Test retrieval quality

**Tasks**:
- [ ] Write `TFIDFRetrieval` class in `retrieval.py`
- [ ] Implement IDF calculation
- [ ] Implement document scoring
- [ ] Write query processing function
- [ ] Test on 5-10 sample queries manually
- [ ] Verify: Retrieval returns reasonable results

**Deliverable**: Working TF-IDF retrieval system

---

#### Week 4: BM25 Retrieval
**Goals**:
- Implement BM25 scoring
- Compare BM25 vs TF-IDF informally

**Tasks**:
- [ ] Write `BM25Retrieval` class in `retrieval.py`
- [ ] Implement BM25 scoring with k1, b parameters
- [ ] Test on same queries as TF-IDF
- [ ] Compare result quality subjectively
- [ ] Tune k1 and b if needed
- [ ] Verify: BM25 generally ranks better than TF-IDF

**Deliverable**: Working BM25 retrieval system

---

#### Week 5: Query Expansion (PRF)
**Goals**:
- Implement pseudo-relevance feedback
- Test expansion effectiveness

**Tasks**:
- [ ] Write `PseudoRelevanceFeedback` class in `query_expansion.py`
- [ ] Implement term extraction from top documents
- [ ] Implement query expansion logic
- [ ] Test on ambiguous queries ("python", "turkey")
- [ ] Compare expanded vs non-expanded results
- [ ] Optional: Implement Rocchio algorithm
- [ ] Verify: Expanded queries return more focused results

**Deliverable**: Working PRF implementation

---

#### Week 6: Evaluation Framework
**Goals**:
- Create test queries
- Label relevance judgments
- Implement evaluation metrics

**Tasks**:
- [ ] Create `queries.json` with 15 test queries
- [ ] Run baseline retrieval on all queries
- [ ] Manually label top-20 results per query (300+ labels)
- [ ] Save to `relevance_judgments.json`
- [ ] Write evaluation metrics in `metrics.py`
- [ ] Test metric calculations
- [ ] Verify: Metrics produce sensible values

**Deliverable**: Labeled test set and evaluation scripts

---

#### Week 7: Experiments and Analysis
**Goals**:
- Run all experimental configurations
- Compare results systematically
- Analyze findings

**Tasks**:
- [ ] Write `run_experiments.py`
- [ ] Run baseline TF-IDF
- [ ] Run expanded TF-IDF
- [ ] Run baseline BM25
- [ ] Run expanded BM25
- [ ] Generate results table
- [ ] Analyze improvements from PRF
- [ ] Analyze ambiguous vs specific queries separately
- [ ] Create visualizations (optional: bar charts)
- [ ] Verify: PRF improves MAP by >10%

**Deliverable**: Experimental results and analysis

---

#### Week 8: Documentation and Presentation
**Goals**:
- Complete project documentation
- Prepare presentation
- Final testing and bug fixes

**Tasks**:
- [ ] Write README.md
- [ ] Complete code comments and docstrings
- [ ] Create presentation slides
- [ ] Prepare demo for presentation
- [ ] Polish CLI interface
- [ ] Final code review and cleanup
- [ ] Verify: All components work end-to-end

**Deliverable**: Complete project with documentation

---

### Critical Path
1. **Weeks 1-2**: Foundation (data + indexing) - cannot proceed without these
2. **Weeks 3-4**: Core retrieval - must work before evaluation
3. **Week 5**: PRF - depends on retrieval working
4. **Week 6**: Evaluation - requires manual labeling (most time-consuming)
5. **Week 7**: Experiments - requires completed evaluation framework
6. **Week 8**: Polish and presentation

---

## 14. Risks and Simplifications

### Things to Avoid (Overcomplication)

1. **Custom Tokenizers**
   - ❌ Don't: Build complex tokenizers with regex patterns for every edge case
   - ✅ Do: Use simple NLTK tokenization or regex `\b[a-z0-9]+\b`

2. **Deep Learning**
   - ❌ Don't: Use transformers (BERT, GPT) for query expansion
   - ✅ Do: Stick to TF-IDF-based PRF

3. **Complex UI**
   - ❌ Don't: Build web interface with Flask/Django
   - ✅ Do: Use simple CLI with argparse

4. **Advanced NLP**
   - ❌ Don't: Use spaCy, named entity recognition, dependency parsing
   - ✅ Do: Use basic NLTK preprocessing

5. **Distributed Systems**
   - ❌ Don't: Use Elasticsearch, Solr, or distributed indexing
   - ✅ Do: In-memory index with pickle serialization

6. **Sophisticated Query Expansion**
   - ❌ Don't: Use WordNet, knowledge graphs, or neural embeddings
   - ✅ Do: Simple term frequency-based PRF

### Practical Shortcuts

1. **Dataset Size**
   - **Full dataset**: 21,578 documents
   - **Shortcut**: Use only 5,000 documents for faster development/testing
   - Trade-off: Results will differ slightly, but methodology remains valid

2. **Relevance Judgments**
   - **Ideal**: Label 100+ queries with 50 results each
   - **Shortcut**: Label 10-15 queries with 20 results each
   - Trade-off: Less statistical power, but sufficient for semester project

3. **Parameter Tuning**
   - **Ideal**: Grid search over k1 ∈ [1.2, 1.5, 2.0], b ∈ [0.5, 0.75, 1.0]
   - **Shortcut**: Use standard values k1=1.5, b=0.75
   - Trade-off: Possibly suboptimal, but saves significant time

4. **Query Expansion Terms**
   - **Ideal**: Optimize number of feedback docs and expansion terms per query
   - **Shortcut**: Fixed values (feedback_docs=5, expansion_terms=10)
   - Trade-off: May not be optimal for all queries

5. **Index Storage**
   - **Ideal**: Disk-based index with efficient retrieval
   - **Shortcut**: Load entire index into memory using pickle
   - Trade-off: High memory usage (~500MB-1GB), but fast retrieval

6. **Evaluation Metrics**
   - **Ideal**: Graded relevance (0-3 scale), nDCG, RBP
   - **Shortcut**: Binary relevance (0/1), P@K, MAP
   - Trade-off: Less nuanced evaluation, but simpler implementation

### Common Pitfalls and Solutions

#### Pitfall 1: Memory Issues
**Problem**: Loading 21K documents + index crashes due to insufficient RAM

**Solution**:
- Process documents in batches during indexing
- Use generators instead of loading all documents at once
- Reduce dataset size to 10K documents if necessary

```python
def batch_index(documents, batch_size=1000):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        process_batch(batch)
```

#### Pitfall 2: Slow Retrieval
**Problem**: Each query takes 5-10 seconds

**Solution**:
- Ensure postings lists are pre-sorted by doc_id
- Use dictionaries for O(1) lookup
- Limit candidate documents (only consider docs containing query terms)

#### Pitfall 3: Poor Retrieval Quality
**Problem**: Top results seem random or irrelevant

**Solution**:
- Verify preprocessing is working (check stemming output)
- Ensure IDF calculation is correct (check for division by zero)
- Validate term frequencies in index
- Test with very specific queries first ("stock market crash")

#### Pitfall 4: PRF Worsens Results
**Problem**: Query expansion reduces precision

**Solution**:
- Reduce number of feedback documents (try 3-5 instead of 10)
- Reduce number of expansion terms (try 5 instead of 10)
- Weight original query higher (alpha=1.5, beta=0.5)
- Filter expansion terms (remove very frequent terms)

#### Pitfall 5: Inconsistent Results
**Problem**: Same query returns different results each run

**Solution**:
- Ensure deterministic tie-breaking in ranking
- Sort by (score, doc_id) instead of just score
- Check for randomness in preprocessing or term selection

### Debugging Tips

1. **Start Small**: Test on 10 documents before scaling to 21K
2. **Inspect Intermediate Outputs**: Print tokens, stemmed terms, postings
3. **Manual Verification**: For a query, manually check if top documents are reasonable
4. **Unit Tests**: Write tests for preprocessing, scoring functions
5. **Logging**: Add logging statements to track execution flow

### Time Management Strategy

**If Behind Schedule**:
- **Week 2**: Use pre-built index library (Whoosh)
- **Week 4**: Skip BM25, focus on TF-IDF only
- **Week 5**: Use simpler PRF (just add top-5 most frequent terms)
- **Week 6**: Reduce query set to 10 queries, label only top-10 results

**If Ahead of Schedule**:
- Implement Rocchio algorithm
- Add more sophisticated term weighting in PRF
- Build simple web UI with Flask
- Add query suggestion feature
- Implement query log analysis

---

## Conclusion

This blueprint provides a complete roadmap for implementing an ambiguity-aware news retrieval system. Follow the week-by-week timeline, implement each component systematically, and avoid overcomplicating the design.

**Key Success Factors**:
1. Start with data loading and preprocessing—get this right first
2. Build and test each component incrementally
3. Manually verify results at each stage
4. Use simple, proven techniques (TF-IDF, BM25, PRF)
5. Focus on evaluation—strong experimental validation is crucial
6. Document as you go—don't save documentation for the end

**Expected Outcomes**:
- Working retrieval system with TF-IDF and BM25
- Query expansion improves MAP by 10-20%
- Clear demonstration of PRF effectiveness on ambiguous queries
- Complete evaluation with relevance judgments and metrics

Good luck with your implementation!

---

## Appendix: Quick Reference

### Installation Commands
```bash
python -m venv venv
source venv/bin/activate
pip install nltk numpy beautifulsoup4 lxml scikit-learn
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Key Formulas

**TF-IDF**:
```
score = Σ tf(t,d) × log(N / df(t))
```

**BM25**:
```
score = Σ idf(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1 × (1 - b + b × (|d| / avgdl)))
```

**Precision@K**:
```
P@K = |relevant ∩ retrieved_at_k| / k
```

**MAP**:
```
MAP = (1/|Q|) × Σ AP(q)
```

### Important Files
- `data/processed/documents.json` - Processed documents
- `data/index/inverted_index.pkl` - Serialized index
- `evaluation/queries.json` - Test queries
- `evaluation/relevance_judgments.json` - Relevance labels
- `src/main.py` - CLI interface
