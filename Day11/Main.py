import numpy as np
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("Error: scikit-learn is not installed.")
    print("Please install it with: pip install scikit-learn")
    exit(1)

# ----------------------------------------
# STEP 1: Input Documents
# ----------------------------------------
doc1 = "Artificial intelligence and machine learning are transforming technology."
doc2 = "Machine learning is a subset of artificial intelligence used in modern technology."

documents = [doc1, doc2]

# ----------------------------------------
# STEP 2: Preprocessing
# ----------------------------------------
def preprocess(text):
    text = text.lower()                     
    text = re.sub(r'[^a-z\s]', '', text)    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

documents = [preprocess(doc) for doc in documents]

print("Preprocessed Documents:")
print("Doc1:", documents[0])
print("Doc2:", documents[1])

# ----------------------------------------
# STEP 3: Vectorization (TF-IDF)
# ----------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

print("\nVocabulary:")
print(feature_names)

vectors = tfidf_matrix.toarray()

print("\nTF-IDF Vectors:")
print("Doc1 Vector:", np.round(vectors[0], 3))
print("Doc2 Vector:", np.round(vectors[1], 3))

# ----------------------------------------
# STEP 4: Manual Cosine Similarity
# ----------------------------------------
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

similarity_score = cosine_similarity(vectors[0], vectors[1])

print("\nCosine Similarity Score:", round(similarity_score, 3))

# ----------------------------------------
# STEP 5: Interpretation
# ----------------------------------------
print("\nInterpretation:")
if similarity_score > 0.75:
    print("High similarity: The documents are very similar in content.")
elif similarity_score > 0.4:
    print("Moderate similarity: The documents share important concepts.")
else:
    print("Low similarity: The documents discuss different topics.")
