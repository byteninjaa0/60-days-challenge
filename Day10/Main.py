from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

documents = [
    "NLP is fun and powerful",
    "NLP enables machines to understand text",
    "Text data needs to be converted into numbers",
    "Machine learning works on numerical data",
    "NLP NLP NLP text text"
]

bow_vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)

bow_matrix = bow_vectorizer.fit_transform(documents)

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n--- Bag-of-Words ---")
print("Vocabulary:")
print(bow_vectorizer.get_feature_names_out())
print("\nBoW Matrix:")
print(bow_matrix.toarray())

print("\n--- TF-IDF ---")
print("Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:")
print(np.round(tfidf_matrix.toarray(), 3))
