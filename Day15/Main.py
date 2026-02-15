import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# -----------------------------------
# Sample Dataset
# -----------------------------------

documents = [
    "Natural language processing is powerful",
    "Machine learning enables intelligent systems",
    "Deep learning is a subset of machine learning",
    "Language models process text data",
    "Artificial intelligence and NLP are related fields"
]


# -----------------------------------
# STEP 1 – Convert Text to Embeddings
# Using TF-IDF as document embeddings
# -----------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

# Document-Term Matrix
X = vectorizer.fit_transform(documents)

# Convert sparse matrix to dense numpy array
embedding_matrix = X.toarray()

print("\n===== EMBEDDING MATRIX =====")
print("Shape:", embedding_matrix.shape)
print("(Documents, Vocabulary Size)")
print(embedding_matrix)


# -----------------------------------
# STEP 2 – Vectorized Dot Product
# -----------------------------------

# Compute similarity matrix using dot product
dot_product_matrix = embedding_matrix @ embedding_matrix.T

print("\n===== DOT PRODUCT MATRIX =====")
print("Shape:", dot_product_matrix.shape)
print(dot_product_matrix)


# -----------------------------------
# STEP 3 – Cosine Similarity (Fully Vectorized)
# -----------------------------------

# Normalize embeddings (L2 norm)
normalized_embeddings = normalize(embedding_matrix)

cosine_similarity_matrix = normalized_embeddings @ normalized_embeddings.T

print("\n===== COSINE SIMILARITY MATRIX =====")
print("Shape:", cosine_similarity_matrix.shape)
print(np.round(cosine_similarity_matrix, 3))


# -----------------------------------
# STEP 4 – Mean Embedding (Vectorized)
# -----------------------------------

mean_embedding = np.mean(embedding_matrix, axis=0)

print("\n===== MEAN EMBEDDING =====")
print("Shape:", mean_embedding.shape)
print(mean_embedding)


# -----------------------------------
# EXPLANATION SECTION
# -----------------------------------

print("\n===== MATRIX SHAPES =====")
print("Embedding Matrix Shape:", embedding_matrix.shape)
print("Dot Product Shape:", dot_product_matrix.shape)
print("Cosine Similarity Shape:", cosine_similarity_matrix.shape)
print("Mean Embedding Shape:", mean_embedding.shape)

print("\n===== COMPLEXITY INSIGHT =====")
print("Vectorized matrix multiplication runs in optimized C (BLAS).")
print("Time complexity ~ O(n^2 * d) for similarity.")
print("Much faster than nested Python loops due to low-level optimization.")
