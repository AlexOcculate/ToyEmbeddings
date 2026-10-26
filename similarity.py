# Line 1:  Import math for square root (needed to compute vector magnitudes).
import math


# Line 3: Compute the dot product of two equal-length vectors.
# The dot product measures "alignment" — how much two vectors point in the same direction.
def dot_product(a, b):
    # Line 6: If vectors have different lengths, they're incompatible — return 0 as a safe default.
    if len(a) != len(b):
        return 0.0
    # Line 8: Sum of element-wise products: a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n].
    # Intuition: if both vectors have large values at the same positions, the dot product is large.
    return sum(a[i] * b[i] for i in range(len(a)))


# Line 11: Compute the magnitude (length) of a vector using the Euclidean norm.
# This is the "distance from origin" — like the length of an arrow in space.
def magnitude(v):
    # Line 14: Square each component, sum them, then take the square root (Pythagoras in N dimensions).
    return math.sqrt(sum(x * x for x in v))


# Line 16: Compute cosine similarity between two vectors.
# Returns a value between -1 and 1 (for non-negative TF-IDF vectors, between 0 and 1).
#   1.0 = identical direction (maximum similarity)
#   0.0 = perpendicular (no shared meaning)
#   -1.0 = opposite direction (maximum dissimilarity — rare with TF-IDF since values are non-negative)
def cosine_similarity(a, b):
    # Line 21: Compute magnitudes of both vectors.
    mag_a = magnitude(a)
    mag_b = magnitude(b)

    # Line 23: If either vector is all zeros (zero magnitude), similarity is undefined.
    # We return 0.0 as a safe default — "no information, no similarity."
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    # Line 26: Cosine similarity = dot_product / (|a| * |b|).
    # This normalizes the dot product by vector lengths, so it measures DIRECTION only,
    # not magnitude. A short arrow and a long arrow pointing the same way still get score 1.0.
    return dot_product(a, b) / (mag_a * mag_b)


# Line 29: Find the K most similar documents to a query vector from a list of candidate vectors.
# Returns a list of (index, similarity_score) tuples, sorted by score descending.
def find_nearest(query_vec, candidate_vecs, k=3):
    # Line 32: If there are no candidates, return empty list immediately.
    if not candidate_vecs:
        return []

    # Line 35: Compute similarity between the query and every candidate.
    scores = []
    for i, cand in enumerate(candidate_vecs):
        # Line 37: Calculate cosine similarity for this pair.
        score = cosine_similarity(query_vec, cand)
        # Line 38: Store (index, score) so we can sort and retrieve later.
        scores.append((i, score))

    # Line 40: Sort by score in descending order (most similar first).
    scores.sort(key=lambda x: x[1], reverse=True)

    # Line 42: Return only the top-k results.
    return scores[:k]


# Line 44: Convert a raw similarity score (0 to 1) into a human-readable label.
# This makes output more intuitive for non-technical readers.
def similarity_label(score):
    # Line 47: Very high similarity — nearly identical content.
    if score >= 0.9:
        return "NEARLY IDENTICAL"
    # Line 49: High similarity — clearly related topics.
    if score >= 0.7:
        return "STRONGLY RELATED"
    # Line 51: Moderate similarity — some shared terms but different focus.
    if score >= 0.4:
        return "MODERATELY RELATED"
    # Line 53: Low similarity — barely any overlap.
    if score >= 0.1:
        return "WEAKLY RELATED"
    # Line 55: Essentially unrelated.
    return "UNRELATED"
