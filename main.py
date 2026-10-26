# Line 1:  Import the EmbeddingEngine class from our local module.
from embedding_engine import EmbeddingEngine

# Line 2:  Import similarity functions from our local module.
from similarity import cosine_similarity, find_nearest, similarity_label, magnitude

# Line 4: Print a header to identify this demo run in terminal output.
print("=" * 60)
# Line 5: Title of the module for clear identification.
print("  TOYAGENTS #5 — ToyEmbeddings")
# Line 6: One-line description shown to the user.
print("  Text → Numbers → Similarity (TF-IDF + Cosine)")
# Line 7: Close the header block.
print("=" * 60)


# Line 9: Define a small corpus of documents for demonstration.
# These are chosen so that some pairs are clearly similar and others are not,
# making the similarity scores intuitive to interpret.
corpus = [
    # Line 12: Doc 0 — about Python programming (keywords: python, code, language, program)
    "Python is a great programming language for writing clean code.",
    # Line 14: Doc 1 — also about Python (should be SIMILAR to doc 0)
    "You can write elegant code in Python for any programming task.",
    # Line 16: Doc 2 — about cooking (completely different topic → should be UNRELATED to docs 0/1)
    "A good recipe requires fresh ingredients and careful preparation.",
    # Line 18: Doc 3 — about machines (different from all above)
    "The machine processes data through a series of mechanical steps.",
    # Line 20: Doc 4 — about Python again but with different phrasing (should be MODERATELY similar to 0/1)
    "Learning to program in Python takes practice and patience.",
]


# Line 22: Create the embedding engine.
engine = EmbeddingEngine()

# Line 23: Fit the engine to our corpus — builds vocabulary and computes IDF weights.
engine.fit(corpus)

# Line 25: Transform all documents into vectors (list of lists of floats).
vectors = [engine.transform(doc) for doc in corpus]


# ─────────────────────────────────────────────────────────────
#  SECTION 1: Show what an embedding actually looks like
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
# Line 32: Section header for the first demonstration.
print("  SECTION 1: What does a text vector look like?")
print("─" * 60)

# Line 34: Show the vocabulary that was built from the corpus.
print(f"\n  Vocabulary size: {engine.dim()} terms")
# Line 35: Print the sorted vocabulary so the user can see what "dimensions" exist.
print(f"  Terms: {engine.vocabulary}")

# Line 36: Display the vector for document 0 (the Python programming doc).
doc0_vec = vectors[0]
# Line 37: Show which terms have non-zero weights and their values.
print(f'\n  Document 0: "{corpus[0]}"')
# Line 38: Explain the top-5 most important terms in this document's vector.
top_terms = engine.explain_vector(corpus[0], top_k=5)
for term, weight in top_terms:
    # Line 40: Print each term with its TF-IDF weight, formatted for readability.
    print(f"    {term:<15} → {weight:.4f}")

# Line 42: Show the full raw vector (truncated if very long) to demonstrate it's just numbers.
print(f"\n  Raw vector (first 10 dims): {doc0_vec[:10]}")
# Line 43: Show the magnitude (length) of the vector — a single number summarizing "how much signal."
print(f"  Vector magnitude: {magnitude(doc0_vec):.4f}")


# ─────────────────────────────────────────────────────────────
#  SECTION 2: Pairwise similarity between all documents
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
# Line 50: Section header for the similarity matrix.
print("  SECTION 2: How similar are these documents to each other?")
print("─" * 60)

# Line 52: Print a header row for the similarity table.
print(f"\n  {'':>12}", end="")
for j in range(len(corpus)):
    # Line 54: Label each column with its document index.
    print(f"  Doc{j:<3}", end="")
print()  # Newline after header row.

# Line 56: Compute and display the full similarity matrix (upper triangle).
for i in range(len(corpus)):
    # Line 58: Label the row with document index.
    print(f"  {'Doc' + str(i):>12}", end="")
    for j in range(len(corpus)):
        # Line 60: Compute cosine similarity between doc i and doc j.
        sim = cosine_similarity(vectors[i], vectors[j])
        # Line 61: Format the score to 3 decimal places, right-aligned in a 7-char field.
        print(f"  {sim:>6.3f}", end="")
    print()  # Newline after each row.

# Line 63: Add a brief interpretation for the user.
print("\n  Reading the matrix:")
# Line 64: Explain what high scores mean in context.
print("    Doc0 vs Doc1 = high  → both about Python programming")
# Line 65: Explain what low scores mean in context.
print("    Doc0 vs Doc2 = low   → programming vs cooking (unrelated)")


# ─────────────────────────────────────────────────────────────
#  SECTION 3: Nearest-neighbor search (the "search engine" demo)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
# Line 72: Section header for the search demonstration.
print("  SECTION 3: Search — find the most similar documents to a query")
print("─" * 60)

# Line 74: Define a new query that wasn't in the corpus but is related to Python docs.
query = "I want to learn how to write software with a popular scripting language."
# Line 75: Embed the query using the same engine (vocabulary was built from corpus).
query_vec = engine.transform(query)

# Line 77: Find the top-3 most similar documents in our corpus.
results = find_nearest(query_vec, vectors, k=3)

# Line 79: Print the query and its results with human-readable labels.
print(f'\n  Query: "{query}"')
print(f"  Top {len(results)} matches:")
for rank, (doc_idx, score) in enumerate(results, start=1):
    # Line 82: Get a human-friendly label for this similarity score.
    label = similarity_label(score)
    # Line 83: Print the rank, document text, score, and interpretation.
    print(f'    {rank}. [Doc{doc_idx}] "{corpus[doc_idx]}"')
    # Line 84: Show the numeric score and its qualitative label side by side.
    print(f"       Score: {score:.4f}  →  {label}")


# ─────────────────────────────────────────────────────────────
#  SECTION 4: Why TF-IDF? (Show that rare words matter more)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
# Line 92: Section header explaining the IDF insight.
print("  SECTION 4: Why TF-IDF? Rare words carry more signal")
print("─" * 60)

# Line 94: Show the IDF weight for a common word vs. a rare word in our corpus.
common_term = "code"  # Appears in docs 0 and 1 (2 out of 5 docs)
rare_term = "preparation"  # Appears in only doc 2 (1 out of 5 docs)

# Line 96: Compute and display the IDF for each, explaining the difference.
idf_common = engine._idf(common_term)
idf_rare = engine._idf(rare_term)

# Line 98: Print comparison showing why rare terms get higher weights.
print(
    f"\n  Term '{common_term}' appears in {engine.doc_freq.get(common_term, 0)}/{engine.doc_count} docs"
)
print(f"    IDF weight: {idf_common:.4f}")
print(
    f"  Term '{rare_term}' appears in {engine.doc_freq.get(rare_term, 0)}/{engine.doc_count} docs"
)
print(f"    IDF weight: {idf_rare:.4f}")
# Line 101: Explain the intuition in plain language.
print(f"\n  Insight: '{rare_term}' gets a HIGHER weight because it's rarer.")
print("  If a word appears in every document, it can't help you tell them apart.")


# Line 103: Final summary line to close the demo.
print("\n" + "=" * 60)
print("  KEY INSIGHT: An embedding is just a list of numbers.")
print("  Similarity is just a dot product divided by magnitudes.")
print("  No GPU. No model. No API. Just math on word counts.")
print("=" * 60)
