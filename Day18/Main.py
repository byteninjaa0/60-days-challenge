import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.optimize import minimize


# -----------------------------------
# STEP 1 – Create Text Representations
# -----------------------------------

documents_A = [
    "Natural language processing is powerful",
    "Machine learning enables intelligent systems",
    "Deep learning models process text"
]

documents_B = [
    "NLP is a strong field",
    "Intelligent systems use machine learning",
    "Text is processed by deep models"
]

vectorizer = TfidfVectorizer(stop_words="english")

X_A = vectorizer.fit_transform(documents_A).toarray()
X_B = vectorizer.transform(documents_B).toarray()

# Normalize embeddings for cosine similarity
X_A = normalize(X_A)
X_B = normalize(X_B)

print("\n===== INITIAL EMBEDDING SHAPES =====")
print("X_A Shape:", X_A.shape)
print("X_B Shape:", X_B.shape)


# -----------------------------------
# STEP 2 – Define Cost Function
# -----------------------------------

# We learn a transformation matrix W
# Goal: minimize difference between X_A @ W and X_B

n_samples, dim = X_A.shape

def cost_function(W_flat):
    W = W_flat.reshape(dim, dim)
    transformed = X_A @ W
    loss = np.mean((transformed - X_B) ** 2)
    return loss


# -----------------------------------
# STEP 3 – Optimization
# -----------------------------------

# Initialize W as identity
W_initial = np.eye(dim).flatten()

print("\n===== INITIAL COST =====")
initial_cost = cost_function(W_initial)
print("Initial Loss:", initial_cost)

result = minimize(
    cost_function,
    W_initial,
    method="L-BFGS-B",
    options={"maxiter": 200}
)

W_optimized = result.x.reshape(dim, dim)

print("\n===== OPTIMIZATION STATUS =====")
print("Success:", result.success)
print("Iterations:", result.nit)


# -----------------------------------
# STEP 4 – Evaluate After Optimization
# -----------------------------------

final_cost = cost_function(result.x)

print("\n===== FINAL COST =====")
print("Final Loss:", final_cost)

print("\nLoss Reduction:", initial_cost - final_cost)


# -----------------------------------
# STEP 5 – Similarity Evaluation
# -----------------------------------

def cosine_similarity_matrix(A, B):
    return A @ B.T

similarity_before = cosine_similarity_matrix(X_A, X_B)

transformed_A = X_A @ W_optimized
similarity_after = cosine_similarity_matrix(transformed_A, X_B)

print("\n===== SIMILARITY BEFORE OPTIMIZATION =====")
print(np.round(similarity_before, 3))

print("\n===== SIMILARITY AFTER OPTIMIZATION =====")
print(np.round(similarity_after, 3))


# -----------------------------------
# EXPLANATION SECTION
# -----------------------------------

print("\n===== INTERPRETATION =====")
print("Cost function: Mean Squared Error between aligned embeddings.")
print("Objective: Minimize distance between transformed X_A and X_B.")
print("Optimization method: L-BFGS-B (quasi-Newton method).")
print("Matrix Shape W:", W_optimized.shape)
print("Transformation improves similarity alignment.")
