# ToyEmbeddings — Module #5

Convert text into numeric vectors where similar texts produce similar vectors.

**Prerequisite:** understand ToyPlanner (Module #4)

## What Is This?

An "embedding" is a way to represent text as a list of numbers so that computers can measure how similar two pieces of text are using basic arithmetic. This module builds a TF-IDF embedding engine from scratch — no GPU, no model weights, no API calls. You'll see that the magic of "text similarity" is just counting words and measuring angles.

## Folder Structure

    ToyEmbeddings/
    ├── main.py                # Demo: embed texts, compute similarity, search
    ├── embedding_engine.py    # TF-IDF vectorization engine
    ├── similarity.py          # Cosine similarity + nearest-neighbor search
    ├── test_adversarial.py    # 5 adversarial tests
    └── README.md              # This file

## Dependencies

**External:** None (pure Python standard library)

**Stdlib used:** `math` (for log and sqrt)

**Internal import graph:**
- `main.py` → imports `embedding_engine`, `similarity`
- `test_adversarial.py` → imports `embedding_engine`, `similarity`
- `embedding_engine.py` → imports `math` only
- `similarity.py` → imports `math` only

## Execution

    cd ToyEmbeddings
    python main.py
    python test_adversarial.py

## Expected Output

See the terminal output when running `python main.py`. The demo shows:
1. What a vector looks like (term → weight mapping)
2. A pairwise similarity matrix
3. A nearest-neighbor search over the corpus
4. Why rare words get higher weights (IDF explanation)

## Architecture Diagram

    ┌─────────────────────────────────────────────────────────────┐
    │                        main.py                              │
    │  (demo driver: defines corpus, runs sections 1-4)           │
    └──────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
    ┌─────────────────────┐       ┌─────────────────────────┐
    │  embedding_engine   │       │     similarity.py       │
    │                     │       │                         │
    │  • tokenize()       │       │  • dot_product(a, b)    │
    │  • fit(corpus)      │       │  • magnitude(v)         │
    │  • transform(text)  │       │  • cosine_similarity()  │
    │  • explain_vector() │       │  • find_nearest()       │
    └─────────────────────┘       │  • similarity_label()   │
                                  └─────────────────────────┘

    Data flow:
    text → tokenize → TF counts × IDF weights → vector (list of floats)
    vector A + vector B → dot_product / (|A| × |B|) → similarity score [0..1]

---
---
---


# Module #5: ToyEmbeddings — Turn 2 (Diagrams)

---

## A. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              main.py (Driver)                                       │
│                                                                                     │
│   Defines corpus → creates engine → runs 4 demo sections                            │
└──────────┬──────────────────────────────────────────────────────────┬───────────────┘
           │                                                          │
           │ calls                                                    │ calls
           ▼                                                          ▼
┌────────────────────────────────────┐     ┌────────────────────────────────────────┐
│      embedding_engine.py           │     │          similarity.py                 │
│                                    │     │                                        │
│  ┌──────────────────────────────┐  │     │  ┌──────────────────────────────────┐  │
│  │  tokenize(text)              │  │     │  │  dot_product(a, b)               │  │
│  │  • lowercase                 │  │     │  │  • element-wise multiply + sum   │  │
│  │  • strip non-alnum           │  │     │  └──────────────────────────────────┘  │
│  │  • split on whitespace       │  │     │                                        │
│  │  • filter STOP_WORDS         │  │     │  ┌──────────────────────────────────┐  │
│  └──────────────┬───────────────┘  │     │  │  magnitude(v)                    │  │
│                 │                  │     │  │  • sqrt(sum of squares)          │  │
│                 ▼                  │     │  └──────────────┬───────────────────┘  │
│  ┌──────────────────────────────┐  │     │                 │                      │
│  │  fit(corpus)                 │  │     │                 │ used by              │
│  │  • build vocabulary (sorted) │  │     │                 ▼                      │
│  │  • count doc_freq per term   │  │     │  ┌──────────────────────────────────┐  │
│  └──────────────┬───────────────┘  │     │  │  cosine_similarity(a, b)         │  │
│                 │                  │     │  │  • dot / (|a| × |b|)             │  │
│                 ▼                  │     │  │  • returns float in [0, 1]       │  │
│  ┌──────────────────────────────┐  │     │  └──────────────┬───────────────────┘  │
│  │  transform(text)             │  │     │                 │                      │
│  │  • tokenize                  │  │     │                 │ used by              │
│  │  • compute TF (proportions)  │  │     │                 ▼                      │
│  │  • compute IDF per term      │  │     │  ┌──────────────────────────────────┐  │
│  │  • assemble vector           │  │     │  │  find_nearest(query, cands, k)   │  │
│  │    [tf[i] × idf[i]]          │  │     │  │  • score all candidates          │  │
│  └──────────────┬───────────────┘  │     │  │  • sort descending               │  │
│                 │                  │     │  │  • return top-k (idx, score)     │  │
│                 ▼                  │     │  └──────────────────────────────────┘  │
│  ┌──────────────────────────────┐  │     │                                        │
│  │  explain_vector(text, k)     │  │     │  ┌──────────────────────────────────┐  │
│  │  • transform → sort → top-k  │  │     │  │  similarity_label(score)         │  │
│  └──────────────────────────────┘  │     │  │  • maps [0,1] to human label     │  │
│                                    │     │  └──────────────────────────────────┘  │
└────────────────────────────────────┘     └────────────────────────────────────────┘

    DATA FLOW (left to right):

    ┌────────┐      ┌───────────┐      ┌─────────────┐      ┌──────────────────┐
    │  TEXT  │ ───► │ TOKENS    │ ───► │ TF × IDF    │ ───► │  VECTOR          │
    │ (raw)  │      │ (list of  │      │ (per-term   │      │  (list of floats │
    │        │      │  strings) │      │  weights)   │      │   length = vocab │
    └────────┘      └───────────┘      └─────────────┘      └────────┬─────────┘
                                                                      │
                                                                      ▼
                                                          ┌───────────────────┐
                                                          │  SIMILARITY SCORE │
                                                          │  (float 0.0–1.0)  │
                                                          └───────────────────┘

    PRODUCTION ANALOGY:

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  This module (toy)          │  Production equivalent                        │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │  EmbeddingEngine            │  Sentence-BERT / OpenAI / Cohere API          │
    │  (TF-IDF, ~24 dims)        │  (neural network, 384–1536 dims)               │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │  find_nearest()             │  Pinecone / Weaviate / FAISS                  │
    │  (brute-force O(n))        │  (HNSW index, O(log n))                        │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │  cosine_similarity()        │  Same math — just optimized in C/CUDA         │
    └──────────────────────────────┴──────────────────────────────────────────────┘

    LEGEND:
    ┌─────────────────┐   A module/file (box with internal components)
    │  component      │
    │  method()       │   A function or method within the module
    └─────────────────┘
    ───►              Data or call flow direction
    │                 Vertical dependency (caller → callee)

    KEY DESIGN DECISIONS:
    1. embedding_engine.py and similarity.py are SEPARATE because they solve
       different problems: "text → numbers" vs "numbers → relationship."
       In production, these map to an embedding API and a vector database —
       two completely separate services.

    2. The engine is STATEFUL (it remembers vocabulary and doc_freq after fit()).
       This mirrors real embedding models: you train/fit once, then transform
       many times. You can't embed text without first knowing the "space."

    3. find_nearest() is deliberately brute-force (O(n) comparisons).
       At n=5 this is fine. At n=10M you'd need an index — that's Module #6.
```

---

## B. Sequence Diagram

Shows the exact call sequence when `main.py` runs its full demo:

```
  main.py          EmbeddingEngine         similarity.py          (Internal math)
    │                    │                      │                        │
    │  ┌───────────────┐ │                      │                        │
    │  │ Create engine │ │                      │                        │
    │  └───────┬───────┘ │                      │                        │
    │          │         │                      │                        │
    │          │ 1. fit(corpus)                 │                        │
    │──────────┼────────►│                      │                        │
    │          │         │                      │                        │
    │          │         │  ┌─────────────────────────────────────────┐  │
    │          │         │  │ PHASE A: Build Vocabulary               │  │
    │          │         │  │                                         │  │
    │          │         │  │ 2. For each doc in corpus:              │  │
    │          │         │  │     tokenize(doc) ───────────────────►──┼──┤
    │          │         │  │       • lowercase                       │  │
    │          │         │  │       • strip non-alnum                 │  │
    │          │         │  │       • split                           │  │
    │          │         │  │       • remove stop words               │  │
    │          │         │  │     add tokens to all_terms set         │  │
    │          │         │  │                                         │  │
    │          │         │  │ 3. Sort all_terms → self.vocabulary     │  │
    │          │         │  │ 4. Build term_to_index dict             │  │
    │          │         │  └─────────────────────────────────────────┘  │
    │          │         │                                               │
    │          │         │  ┌─────────────────────────────────────────┐  │
    │          │         │  │ PHASE B: Compute Document Frequencies   │  │
    │          │         │  │                                         │  │
    │          │         │  │ 5. For each doc in corpus:              │  │
    │          │         │  │     tokenize(doc) → unique_tokens set   │  │
    │          │         │  │     for t in unique_tokens:             │  │
    │          │         │  │       doc_freq[t] += 1                  │  │
    │          │         │  └─────────────────────────────────────────┘  │
    │          │         │                                               │
    │          │         │  6. return self (engine is now FITTED)        │
    │◄─────────┼─────────│                      │                        │
    │          │         │                      │                        │
    │          │ 7. transform(doc_0)            │                        │
    │──────────┼────────►│                      │                        │
    │          │         │                      │                        │
    │          │         │  ┌─────────────────────────────────────────┐  │
    │          │         │  │ PHASE: Single Text → Vector             │  │
    │          │         │  │                                         │  │
    │          │         │  │ 8. tokenize(text)                       │  │
    │          │         │  │     → [token1, token2, ...]             │  │
    │          │         │  │                                         │  │
    │          │         │  │ 9. Compute TF dict:                     │  │
    │          │         │  │     tf[t] = count(t) / total_tokens     │  │
    │          │         │  │                                         │  │
    │          │         │  │ 10. For each term t in tf:              │  │
    │          │         │  │      idf_val = log((N+1)/(df[t]+1)) ───►┼──┤
    │          │         │  │      weight = tf[t] × idf_val           │  │
    │          │         │  │      vector[index_of_t] = weight        │  │
    │          │         │  │                                         │  │
    │          │         │  │ 11. Return vector (list of floats)      │  │
    │          │         │  └─────────────────────────────────────────┘  │
    │◄─────────┼─────────│                      │                        │
    │          │         │                      │                        │
    │          │ 8-11. (repeated for doc_1, doc_2, ..., doc_4)           │
    │──────────┼────────►│                      │                        │
    │◄─────────┼─────────│  (vectors[0..4] now exist)                    │
    │          │         │                      │                        │
    │          │         │ 12. cosine_similarity(vectors[0], vectors[1]) │
    │──────────┼─────────┼─────────────────────►│                        │
    │          │         │                      │                        │
    │          │         │                      │  ┌─────────────────┐   │
    │          │         │                      │  │ 13. dot_product │   │
    │          │         │                      │  │    = Σ a[i]×b[i]│   │
    │          │         │                      │  │                 │   │
    │          │         │                      │  │ 14. magnitude   │   │
    │          │         │                      │  │     = √(Σ a²)   │   │
    │          │         │                      │  │                 │   │
    │          │         │                      │  │ 15. result =    │   │
    │          │         │                      │  │     dot/(|a||b|)│   │
    │          │         │                      │  └────────┬────────┘   │
    │          │         │                      │           │            │
    │          │         │                      │◄──────────┘            │
    │◄─────────┼─────────┼──────────────────────│  16. return score      │
    │          │         │                      │                        │
    │          │         │ 17. find_nearest(query_vec, vectors, k=3)     │
    │──────────┼─────────┼─────────────────────►│                        │
    │          │         │                      │                        │
    │          │         │                      │  ┌─────────────────┐   │
    │          │         │                      │  │ 18. For each    │   │
    │          │         │                      │  │     candidate:  │   │
    │          │         │                      │  │       cosine_   │   │
    │          │         │                      │  │       similarity│   │
    │          │         │                      │  │                 │   │
    │          │         │                      │  │ 19. Sort by     │   │
    │          │         │                      │  │     score DESC  │   │
    │          │         │                      │  │                 │   │
    │          │         │                      │  │ 20. Return top-k│   │
    │          │         │                      │  └────────┬────────┘   │
    │          │         │                      │           │            │
    │          │         │                      │◄──────────┘            │
    │◄─────────┼─────────┼──────────────────────│  21. return [(idx,     │
    │          │         │                      │      score), ...]      │
    │          │         │                      │                        │
    │  22. Print results with similarity_label(score)                    │
    │      (e.g., "MODERATELY RELATED")                                  │
    ▼
```

**Legend:**

| Symbol         | Meaning                                        |
| -------------- | ---------------------------------------------- |
| `───►`         | Method call (synchronous, blocks until return) |
| `◄────`        | Return value                                   |
| `┌──┐`         | Internal processing block (no external call)   |
| Numbered steps | Exact execution order                          |

**Key Observations:**

1. **fit() is called ONCE** per corpus. It's expensive (O(corpus_size × avg_doc_length)) but you do it once.
2. **transform() is called MANY times** — once per document, once per query. It's fast because vocabulary and IDF are already computed.
3. **cosine_similarity() is called O(n) times** inside find_nearest(). For n=5 that's trivial. For n=10M you'd need an index (Module #6).
4. **No loops, no state machines, no agent.** The entire "intelligence" is: count words → multiply by rarity → compare angles.

---

## C. State Machine Diagram

Since Module #5 has **no agent loop**, the "state machine" describes the **lifecycle of a document as it flows through the embedding pipeline**, plus the **engine's readiness states**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ENGINE LIFECYCLE (EmbeddingEngine)                           │
└─────────────────────────────────────────────────────────────────────────────────┘


    ┌───────────┐         fit(corpus)          ┌───────────┐
    │           │  ─────────────────────────►  │           │
    │ UNFITTED  │                              │  FITTED   │
    │           │                              │           │
    │ vocab=[]  │                              │ vocab=[..]│
    │ doc_freq={}│                             │ doc_freq= │
    │ dim()=0   │                              │   {t: n}  │
    └─────┬─────┘                              └─────┬─────┘
          │                                          │
          │ transform(text)                          │ transform(text)
          │ (auto-fits on [text])                    │ (uses existing vocab)
          ▼                                          ▼
    ┌───────────┐                              ┌───────────┐
    │           │                              │           │
    │ AUTO-FIT  │                              │ TRANSFORM │
    │           │                              │           │
    │ fits on   │                              │ tokenize  │
    │ single    │                              │ → TF      │
    │ doc       │                              │ → IDF     │
    └─────┬─────┘                              │ → vector  │
          │                                    └─────┬─────┘
          ▼                                          │
    ┌───────────┐                                    │
    │  FITTED   │ ◄──────────────────────────────────┘
    │ (1-doc)   │      returns vector
    └───────────┘

    GUARDS:
    • UNFITTED + transform() → triggers auto-fit on single doc (safe fallback)
    • FITTED + fit(new_corpus) → RESETS vocabulary and doc_freq (re-fits)
    • Any state + dim() → returns len(vocabulary) (0 if unfitted)


┌─────────────────────────────────────────────────────────────────────────────────┐
│              DOCUMENT PIPELINE (single text through transform())                │
└─────────────────────────────────────────────────────────────────────────────────┘


    ┌──────────┐    tokenize()    ┌──────────────┐   filter stop   ┌──────────────┐
    │          │  ─────────────►  │              │  ───────────►   │              │
    │  RAW     │                  │ TOKENIZED    │                 │  FILTERED    │
    │  TEXT    │                  │ (all tokens) │                 │  TOKENS      │
    │          │                  │              │                 │              │
    │ "Python  │                  │ ["python",   │                 │ ["python",   │
    │  is a    │                  │  "is", "a",  │                 │"programming",│
    │  great   │                  │  "great",    │                 │  "language"] │
    │  program-│                  │  "program-   │                 │              │
    │  ming..." │                 │  "ming", ...]│                 │ (stop words  │
    └──────────┘                  └──────────────┘                 │  removed)    │
                                                                   └──────┬───────┘
                                                                          │
                                                                          │ compute TF
                                                                          ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                                                                              │
    │   ┌──────────────┐   for each term:   ┌──────────────────┐                   │
    │   │  FILTERED    │  tf[t] = count/total│  TF COMPUTED     │                  │
    │   │  TOKENS      │  ───────────────►  │                  │                   │
    │   │              │                     │ {"python": 0.2,  │                  │
    │   │ [t1, t2, ...]│                     │  "programming":  │                  │
    │   └──────────────┘                     │  0.2, ...}       │                  │
    │                                        └────────┬─────────┘                  │
    │                                                 │                            │
    │                                                 │ multiply by IDF            │
    │                                                 ▼                            │
    │                                        ┌──────────────────┐                  │
    │                                        │  TF-IDF WEIGHTS  │                  │
    │                                        │                  │                  │
    │                                        │ {"python":       │                  │
    │                                        │   0.2 × 1.79,    │                  │
    │                                        │  "programming":  │                  │
    │                                        │   0.2 × 1.79,    │                  │
    │                                        │  ...}            │                  │
    │                                        └────────┬─────────┘                  │
    │                                                 │                            │
    │                                                 │ scatter into vector        │
    │                                                 ▼                            │
    │   ┌──────────────┐                    ┌──────────────────┐                   │
    │   │  VECTOR      │ ◄───────────────── │  ASSEMBLING      │                   │
    │   │  COMPLETE    │                    │                  │                   │
    │   │              │                    │ vector[vocab_idx]│                   │
    │   │ [0.0, 0.36,  │                    │ = tf_idf_weight  │                   │
    │   │  0.0, ...]   │                    └──────────────────┘                   │
    │   │              │                                                           │
    │   │ len = vocab  │                                                           │
    │   │ size         │                                                           │
    │   └──────────────┘                                                           │
    │                                                                              │
    └──────────────────────────────────────────────────────────────────────────────┘

    EDGE CASE TRANSITIONS:

    RAW TEXT ──► TOKENIZED ──► FILTERED
                                    │
                                    │ if len(filtered) == 0:
                                    ▼
                              ┌──────────────┐
                              │  ZERO VECTOR │  (all stop words / empty / punctuation)
                              │  [0.0, ...]  │  → cosine with this = 0.0 (safe)
                              └──────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│              SEARCH PIPELINE (find_nearest())                                    │
└─────────────────────────────────────────────────────────────────────────────────┘


    ┌────────────┐      embed query       ┌────────────┐
    │  QUERY     │  ───────────────────►  │  QUERY     │
    │  TEXT      │   engine.transform()   │  VECTOR    │
    └────────────┘                        └─────┬──────┘
                                                │
                                                │ for each candidate:
                                                ▼
                              ┌─────────────────────────────────┐
                              │         SCORING                 │
                              │                                 │
                              │  score[i] = cosine(             │
                              │      query_vec,                 │
                              │      candidates[i]              │
                              │  )                              │
                              │                                 │
                              │  scores = [                     │
                              │    (0, 0.577),                  │
                              │    (1, 0.485),                  │
                              │    (2, 0.000),                  │
                              │    (3, 0.000),                  │
                              │    (4, 0.577)                   │
                              │  ]                              │
                              └────────────────┬────────────────┘
                                               │
                                               │ sort by score DESC
                                               ▼
                              ┌─────────────────────────────────┐
                              │         SORTED                  │
                              │                                 │
                              │  [(0, 0.577),                   │
                              │   (4, 0.577),                   │
                              │   (1, 0.485),                   │
                              │   (2, 0.000),                   │
                              │   (3, 0.000)]                   │
                              └────────────────┬────────────────┘
                                               │
                                               │ take first k
                                               ▼
                              ┌─────────────────────────────────┐
                              │         TOP-K RESULT            │
                              │                                 │
                              │  [(0, 0.577),                   │
                              │   (4, 0.577),                   │
                              │   (1, 0.485)]                   │
                              │                                 │
                              │  → "Doc0 and Doc4 are most      │
                              │     similar to the query"       │
                              └─────────────────────────────────┘

    GUARDS:
    • candidates is empty list → return [] immediately (no crash)
    • query_vec is zero vector → all scores = 0.0 (no division by zero)
    • candidate is zero vector → its score = 0.0 (safe)
    • k > len(candidates) → return all candidates (no IndexError)


┌─────────────────────────────────────────────────────────────────────────────────┐
│                         STATE SUMMARY TABLE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

  ENGINE STATES:
  ┌────────────┬────────────────────────────────────────────────────────────────┐
  │ State      │ Properties                                                     │
  ├────────────┼────────────────────────────────────────────────────────────────┤
  │ UNFITTED   │ vocabulary=[], doc_freq={}, dim()=0                            │
  │            │ transform() auto-fits on single input                          │
  ├────────────┼────────────────────────────────────────────────────────────────┤
  │ FITTED     │ vocabulary=[t1..tn], doc_freq={t:count}, dim()=n               │
  │            │ transform() uses existing vocab + precomputed IDF              │
  └────────────┴────────────────────────────────────────────────────────────────┘

  DOCUMENT PIPELINE STATES:
  ┌──────────────────┬──────────────────────────────────────────────────────────┐
  │ State            │ Properties                                               │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ RAW_TEXT         │ Original string, any encoding/punctuation                │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ TOKENIZED        │ List of lowercase alphanumeric tokens                    │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ FILTERED         │ Tokens with stop words removed (may be empty)            │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ TF_COMPUTED      │ Dict: term → proportion in this doc                      │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ IDF_WEIGHTED     │ Dict: term → tf × idf (the final weight)                 │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ VECTOR_COMPLETE  │ List of floats, length = vocab size                      │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ ZERO_VECTOR      │ All elements are 0.0 (degenerate input)                  │
  └──────────────────┴──────────────────────────────────────────────────────────┘

  SEARCH PIPELINE STATES:
  ┌──────────────────┬──────────────────────────────────────────────────────────┐
  │ State            │ Properties                                               │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ QUERY_EMBEDDED   │ Query text transformed to vector                         │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ SCORING          │ Cosine computed for every candidate                      │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ SORTED           │ All (index, score) pairs in descending order             │
  ├──────────────────┼──────────────────────────────────────────────────────────┤
  │ TOP_K_RETURNED   │ First k entries extracted as final result                │
  └──────────────────┴──────────────────────────────────────────────────────────┘


  KEY DESIGN DECISIONS (annotated on diagrams):

  1. NO LOOP: Unlike modules #0–#4, there is no iterative agent loop.
     The pipeline is strictly linear: text in → numbers out → score out.
     This is intentional — embeddings are a DATA TRANSFORMATION, not a
     decision-making process.

  2. STATEFUL ENGINE: The engine must be fit() before it can transform()
     meaningfully. This mirrors real embedding models (you load weights once,
     then embed many texts). The auto-fit fallback in UNFITTED state is a
     convenience for quick demos, not the intended production pattern.

  3. BRUTE-FORCE SEARCH: find_nearest() compares against ALL candidates.
     This is O(n) per query. At n=5 it's instant. At n=10M you need an
     approximate nearest-neighbor index (HNSW, IVF) — that's Module #6.

  4. ZERO-VECTOR SAFETY: Every function handles the "all zeros" case explicitly.
     In production, this happens when a document contains only stop words or
     is empty. Without explicit guards, cosine_similarity would divide by zero.
```

---

**Module #5 diagrams complete.** When you're ready for the next module, say **"Build module #6 ToyVectorDB"**.


---
---
---



## File Reference

**embedding_engine.py** — The core transformation engine. `fit()` builds a vocabulary from a corpus and computes document frequencies. `transform()` converts any text into a fixed-length numeric vector using TF-IDF weighting. Think of it as a "text-to-numbers translator" that knows which words are important (rare = important) and which are noise (common = ignore).

**similarity.py** — Pure math functions for measuring vector relationships. `cosine_similarity()` answers "do these two texts point in the same direction?" `find_nearest()` is a simple brute-force search: compare the query against every stored vector, sort by score, return top-K. This is exactly what Pinecone/Weaviate do internally — they just add indexing for speed at scale.

**main.py** — The demonstration script. It walks through four sections: (1) what a vector looks like, (2) pairwise similarity matrix, (3) nearest-neighbor search, (4) why IDF matters. Each section builds on the previous one to tell a complete story.

**test_adversarial.py** — Five tests proving the system survives: garbage unicode input, empty/oversized text, zero-magnitude vectors in search, 10,000-document corpus (no hang), and degenerate inputs (single chars, all stop words, punctuation only).

## What This Teaches

| Concept                    | Where to Look                                | Key Insight                                                                                                               |
| -------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Tokenization               | `embedding_engine.py:tokenize()`             | Text must be normalized before counting. Stop words are noise.                                                            |
| Term Frequency             | `embedding_engine.py:transform()`            | Raw counts → proportions. Longer docs don't dominate.                                                                     |
| Inverse Document Frequency | `embedding_engine.py:_idf()`                 | Rare terms are more discriminative. "the" tells you nothing.                                                              |
| Cosine Similarity          | `similarity.py:cosine_similarity()`          | Measures direction, not magnitude. Two texts can differ in length but still be "about the same thing."                    |
| Nearest-Neighbor Search    | `similarity.py:find_nearest()`               | Brute-force: compare against everything, sort, return top-K. Production DBs add indexes for speed — concept is identical. |
| Zero-vector edge case      | `test_adversarial.py:test_module_specific()` | Empty/stopword-only text produces zero vectors. Cosine of zero is undefined → we define it as 0.                          |

## Parent Curriculum

Part of the TOYAGENTS series: https://github.com/AlexOcculate/TOYAGENTS
