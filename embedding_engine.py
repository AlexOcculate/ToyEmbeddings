# Line 1:  Import math for logarithm (used in IDF calculation).
import math

# Line 2:  Define a set of common English stop words to filter out during tokenization.
# These are words that appear so frequently they carry little distinguishing meaning.
STOP_WORDS = frozenset(
    {
        # Line 3:  Articles and determiners (the, a, an) add no topical signal.
        "the",
        "a",
        "an",
        # Line 4:  Prepositions (in, on, at, etc.) are structural, not semantic.
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        # Line 5:  Pronouns (it, he, she, etc.) refer to entities but aren't topical.
        "it",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        # Line 6:  Auxiliary verbs and common function words.
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        # Line 7:  Conjunctions and discourse markers.
        "so",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "as",
        "by",
        "from",
        "with",
        "about",
        "into",
        "through",
        # Line 8:  Fillers and intensifiers that don't change topic.
        "very",
        "really",
        "just",
        "also",
        "too",
        "not",
        "no",
        "yes",
        "some",
        "any",
        "all",
        "each",
        "every",
        "such",
        "only",
    }
)


# Line 10: Define the EmbeddingEngine class that converts text to numeric vectors.
class EmbeddingEngine:
    # Line 11: Constructor accepts an optional pre-built vocabulary (list of terms).
    # If None, vocabulary is built lazily from the first corpus passed to fit().
    def __init__(self, vocabulary=None):
        # Line 13: Store the vocabulary as a list; empty list means "not yet fitted."
        self.vocabulary = list(vocabulary) if vocabulary else []
        # Line 14: Map each term to its index position in the vector (for fast lookup).
        self.term_to_index = {term: i for i, term in enumerate(self.vocabulary)}
        # Line 15: Track the number of documents seen during fit() — needed for IDF.
        self.doc_count = 0
        # Line 16: Track how many documents contain each term (document frequency).
        self.doc_freq = {}

    # Line 18: Tokenize raw text into a list of meaningful terms.
    # This is the "preprocessing" step: normalize, split, filter.
    def tokenize(self, text):
        # Line 20: Convert to lowercase so "The" and "the" are treated identically.
        text = text.lower()
        # Line 21: Replace any character that is NOT a letter or digit with a space.
        # This strips punctuation (commas, periods, quotes) without needing regex.
        cleaned = ""
        for ch in text:
            # Line 23: Keep only alphanumeric characters; replace everything else with space.
            cleaned += ch if ch.isalnum() else " "
        # Line 24: Split on whitespace to get raw tokens.
        tokens = cleaned.split()
        # Line 25: Filter out stop words — they appear in almost every document and dilute signal.
        filtered = [t for t in tokens if t not in STOP_WORDS]
        # Line 26: Return the cleaned list of meaningful terms.
        return filtered

    # Line 28: Fit the engine to a corpus (list of text strings).
    # This builds the vocabulary and computes document frequencies for IDF.
    def fit(self, corpus):
        # Line 30: Reset state in case fit() is called again with a new corpus.
        self.vocabulary = []
        self.term_to_index = {}
        self.doc_count = len(corpus)
        self.doc_freq = {}

        # Line 34: First pass — collect all unique terms across the entire corpus.
        all_terms = set()
        for doc in corpus:
            # Line 36: Tokenize each document and add its terms to the global set.
            tokens = self.tokenize(doc)
            for t in tokens:
                # Line 38: Add to set (duplicates are automatically ignored).
                all_terms.add(t)

        # Line 40: Sort alphabetically for deterministic vector ordering across runs.
        # This ensures the same corpus always produces vectors with terms in the same positions.
        self.vocabulary = sorted(all_terms)
        # Line 42: Rebuild the index map from the sorted vocabulary.
        self.term_to_index = {term: i for i, term in enumerate(self.vocabulary)}

        # Line 44: Second pass — count how many documents contain each term (document frequency).
        # IDF needs this: a term that appears in ALL documents is not discriminative.
        for doc in corpus:
            # Line 46: Use a set per document so each term is counted at most once per doc.
            unique_tokens = set(self.tokenize(doc))
            for t in unique_tokens:
                # Line 48: Increment the document frequency counter for this term.
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        # Line 50: Return self to allow method chaining: engine.fit(corpus).transform(text)
        return self

    # Line 52: Compute the Inverse Document Frequency for a single term.
    # IDF answers: "How rare is this term across all documents?"
    # A term in every document gets low IDF (not useful for distinguishing).
    # A term in only one document gets high IDF (very distinctive).
    def _idf(self, term):
        # Line 56: If the term was never seen during fit(), assign a neutral IDF of 0.
        if term not in self.doc_freq:
            return 0.0
        # Line 58: Standard IDF formula: log(N / df) where N = total docs, df = docs containing term.
        # We add 1 to both numerator and denominator (smoothing) to avoid log(0) if a term
        # somehow appears in zero docs at transform time (shouldn't happen after fit, but safety).
        return math.log((self.doc_count + 1) / (self.doc_freq[term] + 1))

    # Line 62: Convert a single text string into a TF-IDF vector (list of floats).
    # The vector has one position per term in the vocabulary.
    def transform(self, text):
        # Line 65: If no vocabulary exists yet, try to fit on just this single document.
        # This makes the engine usable without an explicit fit() call for quick demos.
        if not self.vocabulary:
            self.fit([text])

        # Line 68: Tokenize the input text into meaningful terms.
        tokens = self.tokenize(text)

        # Line 70: If after filtering stop words there are no tokens left, return a zero vector.
        # This handles edge cases like "the and of" → all stop words → nothing to represent.
        if not tokens:
            return [0.0] * len(self.vocabulary)

        # Line 73: Compute Term Frequency for each token in this document.
        # TF = (count of term in doc) / (total terms in doc).
        # This normalizes so longer documents don't automatically have larger vectors.
        tf = {}
        for t in tokens:
            # Line 75: Count occurrences of each term in this specific document.
            tf[t] = tf.get(t, 0) + 1
        # Line 76: Normalize counts by total token count to get relative frequency.
        total = len(tokens)
        for t in tf:
            # Line 78: Divide raw count by total to get proportion (e.g., 3 out of 10 = 0.3).
            tf[t] = tf[t] / total

        # Line 80: Build the output vector, one slot per vocabulary term.
        vector = [0.0] * len(self.vocabulary)
        for t in tf:
            # Line 82: If this term is in our vocabulary, compute its TF-IDF weight.
            if t in self.term_to_index:
                # Line 83: Multiply term frequency by inverse document frequency.
                # The result: "How important is this term to THIS document, given how rare it is overall?"
                weight = tf[t] * self._idf(t)
                # Line 85: Place the weight at the correct position in the vector.
                vector[self.term_to_index[t]] = weight

        # Line 87: Return the complete numeric vector representing this document.
        return vector

    # Line 89: Convenience method — fit and transform in one call.
    def fit_transform(self, corpus):
        # Line 91: Build vocabulary and IDF from the corpus...
        self.fit(corpus)
        # Line 92: ...then convert each document to its vector representation.
        return [self.transform(doc) for doc in corpus]

    # Line 94: Return the dimensionality of vectors produced by this engine.
    def dim(self):
        # Line 96: The vector length equals the vocabulary size (one slot per known term).
        return len(self.vocabulary)

    # Line 98: Pretty-print a vector with its corresponding terms (for human readability).
    def explain_vector(self, text, top_k=5):
        # Line 100: Transform the text to get its vector.
        vec = self.transform(text)
        # Line 101: Pair each value with its term name.
        pairs = [(self.vocabulary[i], vec[i]) for i in range(len(vec)) if vec[i] > 0]
        # Line 102: Sort by weight descending so the most important terms come first.
        pairs.sort(key=lambda x: x[1], reverse=True)
        # Line 103: Return only the top-k most influential terms and their weights.
        return pairs[:top_k]
