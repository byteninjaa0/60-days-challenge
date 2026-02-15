import re
from collections import Counter
from typing import List

try:
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Warning: NLTK data download issue: {e}")
        print("Attempting to continue anyway...")

    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

except ImportError as e:
    print(f"Error: Required library not installed: {e}")
    print("Please install required packages:")
    print("  pip install nltk scikit-learn numpy")
    exit(1)


# =====================================================
# SAMPLE DATASET
# =====================================================
text_data = """
Natural Language Processing enables machines to understand human language.
NLP is used in chatbots, spam detection, sentiment analysis, and search engines.
Building reusable NLP pipelines improves scalability and performance.
Spam detection models rely heavily on proper text preprocessing.
"""

documents = [text_data]


# =====================================================
# STEP 1 – EASY
# Text Cleaning & Tokenization
# =====================================================

print("========== STEP 1: CLEANING & TOKENIZATION ==========\n")

sentences = sent_tokenize(text_data)
print("Sentence Tokenization:")
print(sentences, "\n")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

cleaned_text = clean_text(text_data)

words = word_tokenize(cleaned_text)

filtered_tokens = [w for w in words if w not in stop_words]

print("Cleaned Tokens:")
print(filtered_tokens)


# =====================================================
# STEP 2 – MEDIUM
# Text Representation & Analysis
# =====================================================

print("\n========== STEP 2: REPRESENTATION & ANALYSIS ==========\n")

# Word Frequency Distribution
freq_dist = Counter(filtered_tokens)

print("Top 20 Frequent Words:")
print(freq_dist.most_common(20), "\n")

# Bag-of-Words
bow_vectorizer = CountVectorizer(stop_words="english")
bow_matrix = bow_vectorizer.fit_transform([cleaned_text])

print("Bag-of-Words Vocabulary:")
print(bow_vectorizer.get_feature_names_out())

print("BoW Vector:")
print(bow_matrix.toarray(), "\n")

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_text])

print("TF-IDF Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())

print("TF-IDF Vector:")
print(np.round(tfidf_matrix.toarray(), 3))


print("\nBoW counts raw frequency.")
print("TF-IDF reduces importance of common words and boosts rare ones.")


# =====================================================
# STEP 3 – HARD
# Reusable NLP Pipeline
# =====================================================

print("\n========== STEP 3: REUSABLE NLP PIPELINE ==========\n")

class NLPPipeline:

    def __init__(self, use_lemmatization=True):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.use_lemmatization = use_lemmatization

    def preprocess(self, texts: List[str]):
        processed = []

        for text in texts:
            text = clean_text(text)
            tokens = word_tokenize(text)
            tokens = [w for w in tokens if w not in self.stop_words]

            if self.use_lemmatization:
                tokens = [self.lemmatizer.lemmatize(w) for w in tokens]

            processed.append(" ".join(tokens))

        return processed


pipeline = NLPPipeline(use_lemmatization=True)

dataset = [
    "Spam detection requires clean text data.",
    "Reusable NLP pipelines help machine learning systems scale.",
    "Proper preprocessing improves model accuracy."
]

processed_dataset = pipeline.preprocess(dataset)

print("Processed Dataset:")
for doc in processed_dataset:
    print(doc)


print("\nScalability & Real-World Use:")
print("This pipeline ensures consistent preprocessing during training and inference.")
print("It reduces duplication and improves maintainability in ML systems.")
