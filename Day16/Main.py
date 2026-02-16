import numpy as np

# -----------------------------------
# STEP 1 – Create Embedding Vectors
# -----------------------------------

# Example: 3 word embeddings, each 4-dimensional
# Shape: (embedding_dim,)
word1 = np.array([0.2, 0.8, 0.5, 0.1])
word2 = np.array([0.3, 0.7, 0.4, 0.2])
word3 = np.array([0.9, 0.1, 0.3, 0.6])

print("\n===== EMBEDDING VECTORS =====")
print("Word1 Shape:", word1.shape)
print("Word2 Shape:", word2.shape)
print("Word3 Shape:", word3.shape)


# -----------------------------------
# STEP 2 – Dot Product Between Vectors
# -----------------------------------

dot_12 = np.dot(word1, word2)
dot_13 = np.dot(word1, word3)

print("\n===== DOT PRODUCTS =====")
print("Dot(word1, word2):", dot_12)
print("Dot(word1, word3):", dot_13)

print("\nInterpretation:")
print("Higher dot product -> more aligned vectors -> more semantic similarity.")


# -----------------------------------
# STEP 3 – Cosine Similarity
# -----------------------------------

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

cos_12 = cosine_similarity(word1, word2)
cos_13 = cosine_similarity(word1, word3)

print("\n===== COSINE SIMILARITY =====")
print("Cosine(word1, word2):", round(cos_12, 3))
print("Cosine(word1, word3):", round(cos_13, 3))

print("\nInterpretation:")
print("Cosine similarity normalizes magnitude and measures angle between vectors.")


# -----------------------------------
# STEP 4 – Create Embedding Matrix
# -----------------------------------

# Stack embeddings into matrix
# Shape: (num_words, embedding_dim)
embedding_matrix = np.vstack([word1, word2, word3])

print("\n===== EMBEDDING MATRIX =====")
print("Shape:", embedding_matrix.shape)
print(embedding_matrix)


# -----------------------------------
# STEP 5 – Matrix Multiplication
# -----------------------------------

# Compute similarity matrix
# (n, d) @ (d, n) -> (n, n)
similarity_matrix = embedding_matrix @ embedding_matrix.T

print("\n===== MATRIX MULTIPLICATION RESULT =====")
print("Shape:", similarity_matrix.shape)
print(similarity_matrix)

print("\nInterpretation:")
print("Diagonal -> dot product of vector with itself (magnitude squared).")
print("Off-diagonal -> pairwise similarity between embeddings.")


# -----------------------------------
# STEP 6 – Matrix Dimension Check Example
# -----------------------------------

print("\n===== DIMENSION CHECK =====")
print("Embedding Matrix Shape:", embedding_matrix.shape)
print("Transpose Shape:", embedding_matrix.T.shape)

print("\nMatrix multiplication rule:")
print("(n, d) x (d, m) -> (n, m)")
print("Inner dimensions must match.")


# -----------------------------------
# COMPLEXITY INSIGHT
# -----------------------------------

print("\n===== WHY MATRIX MULTIPLICATION IS POWERFUL =====")
print("Deep learning layers perform massive matrix multiplications.")
print("Attention mechanisms compute similarity using dot products.")
print("Vectorization allows optimized low-level BLAS execution.")
print("Avoiding Python loops gives massive speed improvements.")
