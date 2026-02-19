import numpy as np

# -----------------------------------
# STEP 1 – Generate Embedding Vectors
# -----------------------------------

np.random.seed(42)

# Simulate high-dimensional embeddings
num_vectors = 5
embedding_dim = 300

embeddings = np.random.randn(num_vectors, embedding_dim)

# Introduce an edge case: zero vector
embeddings[0] = np.zeros(embedding_dim)

print("\n===== ORIGINAL EMBEDDINGS =====")
print("Shape:", embeddings.shape)
print("First vector norm:", np.linalg.norm(embeddings[0]))
print("Second vector norm:", np.linalg.norm(embeddings[1]))


# -----------------------------------
# STEP 2 – Similarity BEFORE Normalization
# -----------------------------------

similarity_before = embeddings @ embeddings.T

print("\n===== DOT PRODUCT SIMILARITY (BEFORE NORMALIZATION) =====")
print(np.round(similarity_before, 3))


# -----------------------------------
# STEP 3 – Numerically Stable L2 Normalization
# -----------------------------------

epsilon = 1e-10  # Small constant for numerical stability

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

# Avoid division by zero
stable_norms = np.where(norms == 0, epsilon, norms)

normalized_embeddings = embeddings / stable_norms

print("\n===== NORMALIZED EMBEDDINGS =====")
print("Norm of first vector:", np.linalg.norm(normalized_embeddings[0]))
print("Norm of second vector:", np.linalg.norm(normalized_embeddings[1]))


# -----------------------------------
# STEP 4 – Similarity AFTER Normalization
# -----------------------------------

similarity_after = normalized_embeddings @ normalized_embeddings.T

print("\n===== COSINE SIMILARITY (AFTER NORMALIZATION) =====")
print(np.round(similarity_after, 3))


# -----------------------------------
# STEP 5 – Interpretation
# -----------------------------------

print("\n===== INTERPRETATION =====")

print("1. Before normalization, dot product depends on vector magnitude.")
print("2. Larger magnitude vectors dominate similarity scores.")
print("3. After normalization, vectors lie on unit hypersphere.")
print("4. Dot product now equals cosine similarity.")
print("5. Zero vectors handled safely using epsilon to avoid division by zero.")

print("\n===== MATRIX SHAPES =====")
print("Embeddings Shape:", embeddings.shape)
print("Normalized Shape:", normalized_embeddings.shape)
print("Similarity Matrix Shape:", similarity_after.shape)

print("\n===== NUMERICAL STABILITY INSIGHT =====")
print("Floating-point division by very small numbers can explode values.")
print("Adding epsilon prevents NaN or Inf results.")
print("Stable normalization is critical in large-scale ML systems.")
