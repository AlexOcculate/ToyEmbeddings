# Line 1:  Import sys and os for path setup (ensures local modules are importable).
import sys, os

# Line 2:  Add the script's directory to the Python path so imports work from any CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Line 4:  Import the core components we'll stress-test.
from embedding_engine import EmbeddingEngine
from similarity import cosine_similarity, find_nearest, magnitude


# ──────────────────────────────────────────────
#  TEST 1: Broken / Chaotic Input
#  Proves: the engine doesn't crash on malformed, garbage, or adversarial text
# ──────────────────────────────────────────────
def test_broken_lm():
    # Line 12: Create a fresh engine for this test.
    engine = EmbeddingEngine()

    # Line 14: Feed a corpus containing garbage: null bytes, unicode symbols, random chars.
    chaotic_corpus = [
        "\x00\x01\x02 binary garbage \xff\xfe",  # Null bytes and high unicode
        "héllo wörld — ünïcödé tëxt 测试 data",  # Mixed unicode scripts
        "!!!???...,,,;;;:::###$$$",  # Only punctuation (no letters/digits)
        "a" * 10000,  # Single repeated char (10k times)
    ]

    # Line 18: Fit should not raise, even with chaotic input.
    engine.fit(chaotic_corpus)

    # Line 20: Transform each document — none should crash.
    for doc in chaotic_corpus:
        vec = engine.transform(doc)
        # Line 22: Assert the vector has the correct dimensionality.
        assert (
            len(vec) == engine.dim()
        ), f"Vector dim mismatch: {len(vec)} != {engine.dim()}"
        # Line 23: Assert all values are finite (no NaN or Infinity from bad math).
        for v in vec:
            assert v == v, "Found NaN in vector"  # NaN != NaN is the standard check
            assert v != float("inf"), "Found Infinity in vector"

    # Line 25: Cosine similarity between two garbage vectors should not crash.
    v1 = engine.transform(chaotic_corpus[0])
    v2 = engine.transform(chaotic_corpus[1])
    sim = cosine_similarity(v1, v2)
    # Line 27: Result must be a valid float in range [-1, 1].
    assert -1.0 <= sim <= 1.0, f"Similarity out of range: {sim}"


# ──────────────────────────────────────────────
#  TEST 2: Invalid / Hostile Input
#  Proves: guardrails handle empty, oversized, and forbidden-pattern inputs
# ──────────────────────────────────────────────
def test_hostile_input():
    # Line 35: Create a fresh engine.
    engine = EmbeddingEngine()

    # Line 37: Fit on a normal small corpus first (gives us a vocabulary).
    engine.fit(["the cat sat on the mat", "dogs play in the park"])

    # Line 39: Test empty string — should produce a zero vector, not crash.
    vec_empty = engine.transform("")
    # Line 41: Zero vector has correct dimensionality and all zeros.
    assert len(vec_empty) == engine.dim()
    assert all(v == 0.0 for v in vec_empty), "Empty string should give zero vector"

    # Line 43: Test a string of only stop words — after filtering, nothing remains.
    vec_stopwords = engine.transform("the and of a in on at to for")
    # Line 45: Should also be a zero vector (all tokens filtered as stop words).
    assert all(
        v == 0.0 for v in vec_stopwords
    ), "All-stopword text should give zero vector"

    # Line 47: Test an extremely long input (1 million characters) — should not hang or crash.
    huge_text = "word " * 500000
    vec_huge = engine.transform(huge_text)
    # Line 49: Should still produce a valid vector of correct size.
    assert len(vec_huge) == engine.dim()

    # Line 51: Cosine similarity with a zero vector must return 0.0, not divide by zero.
    sim_with_zero = cosine_similarity(vec_empty, vec_huge)
    assert (
        sim_with_zero == 0.0
    ), f"Similarity with zero vector should be 0, got {sim_with_zero}"


# ──────────────────────────────────────────────
#  TEST 3: Tool Execution Failure (Zero-Magnitude Vector in Search)
#  Proves: search doesn't crash when query or candidates are zero vectors
# ──────────────────────────────────────────────
def test_tool_failure():
    # Line 60: Create engine and fit on a corpus.
    engine = EmbeddingEngine()
    engine.fit(["hello world", "foo bar baz"])

    # Line 62: Get a valid vector and a zero vector.
    valid_vec = engine.transform("hello world")
    zero_vec = [0.0] * engine.dim()

    # Line 64: Search with a zero query vector — should return results without crashing.
    results = find_nearest(zero_vec, [valid_vec, zero_vec], k=2)
    # Line 65: Should return 2 results (one for each candidate), all with score 0.
    assert len(results) == 2
    for _, score in results:
        # Line 67: All scores should be 0.0 since zero vector has no direction.
        assert score == 0.0, f"Zero vector similarity should be 0, got {score}"

    # Line 69: Search with empty candidate list — should return empty, not crash.
    results_empty = find_nearest(valid_vec, [], k=3)
    assert results_empty == [], "Empty candidate list should return empty results"


# ──────────────────────────────────────────────
#  TEST 4: Infinite Loop / Max Iterations (Very Large Corpus)
#  Proves: the engine completes on large inputs without hanging
# ──────────────────────────────────────────────
def test_infinite_loop():
    # Line 78: Generate a large synthetic corpus (10,000 documents).
    # Each doc uses a rotating set of words to keep vocabulary bounded.
    word_pool = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
    ] * 10
    large_corpus = []
    for i in range(10000):
        # Line 82: Each document is a random-looking combination of pool words.
        doc_words = [word_pool[(i * j) % len(word_pool)] for j in range(20)]
        large_corpus.append(" ".join(doc_words))

    # Line 84: Create engine and fit on the large corpus — must complete without hanging.
    engine = EmbeddingEngine()
    engine.fit(large_corpus)

    # Line 86: Verify vocabulary is bounded (should be at most len(word_pool) unique terms).
    assert engine.dim() <= len(word_pool), f"Vocabulary too large: {engine.dim()}"

    # Line 88: Transform a sample — should work correctly even after fitting on 10k docs.
    vec = engine.transform("alpha beta gamma")
    assert len(vec) == engine.dim()
    # Line 90: At least one value should be non-zero (these words are in the vocabulary).
    assert any(v > 0 for v in vec), "Expected non-zero values for known terms"


# ──────────────────────────────────────────────
#  TEST 5: Module-Specific — Empty text / single char / all stop words
#  Proves: degenerate inputs produce valid (zero) vectors and don't break downstream math
# ──────────────────────────────────────────────
def test_module_specific():
    # Line 99: Create engine with a known corpus.
    engine = EmbeddingEngine()
    engine.fit(["python programming language", "cooking with fresh ingredients"])

    # Line 101: Case A — single character that is NOT a stop word.
    vec_single = engine.transform("x")
    # Line 102: "x" is not in vocabulary (built from the corpus), so it should be all zeros.
    assert all(
        v == 0.0 for v in vec_single
    ), "Unknown single char should give zero vector"

    # Line 104: Case B — single character that IS a stop word.
    vec_stop = engine.transform("a")
    # Line 105: Filtered out as stop word → zero vector.
    assert all(v == 0.0 for v in vec_stop), "Single stop word should give zero vector"

    # Line 107: Case C — text with only punctuation and symbols.
    vec_punct = engine.transform("!!! ??? ... ~~~")
    # Line 108: No alphanumeric tokens survive → zero vector.
    assert all(
        v == 0.0 for v in vec_punct
    ), "Punctuation-only text should give zero vector"

    # Line 110: Case D — verify that cosine_similarity handles the zero-vs-zero case.
    sim_zero_zero = cosine_similarity(vec_single, vec_stop)
    # Line 111: Both are zero vectors → similarity is defined as 0.0 (not NaN, not crash).
    assert sim_zero_zero == 0.0, f"Zero vs zero should be 0.0, got {sim_zero_zero}"

    # Line 113: Case E — verify magnitude of a zero vector is exactly 0.
    assert magnitude(vec_single) == 0.0


# ──────────────────────────────────────────────
#  RUNNER
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Line 120: Define the list of tests with human-readable names.
    tests = [
        ("Broken / Chaotic input", test_broken_lm),
        ("Hostile / Invalid input", test_hostile_input),
        ("Zero-vector search failure", test_tool_failure),
        ("Large corpus (no hang)", test_infinite_loop),
        ("Degenerate text (empty/stop/1ch)", test_module_specific),
    ]
    # Line 126: Track how many tests pass.
    passed = 0
    # Line 127: Iterate through each test, catching exceptions to report failures.
    for name, fn in tests:
        try:
            # Line 129: Execute the test function.
            fn()
            # Line 130: If no exception, mark as passed and print success.
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            # Line 132: If exception, print the failure with the error message.
            print(f"  FAIL  {name}: {e}")
    # Line 134: Print summary line showing pass count vs total.
    print(f"\n  {passed}/{len(tests)} tests passed.")
    # Line 135: Exit with code 1 if any test failed (useful for CI/CD pipelines).
    if passed < len(tests):
        sys.exit(1)
