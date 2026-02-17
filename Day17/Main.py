import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from scipy.stats import poisson, binom
import math

# -----------------------------------
# Sample Corpus
# -----------------------------------

text = """
Natural language processing enables machines to understand human language.
Language models are trained on large datasets.
Machine learning and NLP are closely related fields.
Probability and statistics help analyze word frequencies in text data.
Text data contains common words and rare words.
"""

# -----------------------------------
# STEP 1 – Tokenization
# -----------------------------------

tokens = word_tokenize(text.lower())
tokens = [word for word in tokens if word.isalpha()]

total_words = len(tokens)

print("\n===== TOKENIZATION =====")
print("Total Words:", total_words)
print("First 20 Tokens:", tokens[:20])


# -----------------------------------
# STEP 2 – Word Frequency Counting
# -----------------------------------

freq_dist = Counter(tokens)

print("\n===== WORD FREQUENCY =====")
for word, freq in freq_dist.most_common(10):
    print(word, ":", freq)


# -----------------------------------
# STEP 3 – Probability of Word Occurrence
# -----------------------------------

word_probabilities = {word: freq / total_words for word, freq in freq_dist.items()}

print("\n===== WORD PROBABILITIES =====")
for word, prob in list(word_probabilities.items())[:10]:
    print(word, ":", round(prob, 4))


# -----------------------------------
# STEP 4 – Poisson Distribution Modeling
# -----------------------------------

# Average frequency (λ) across vocabulary
lambda_value = np.mean(list(freq_dist.values()))

print("\n===== POISSON MODEL =====")
print("Lambda (Average Word Frequency):", round(lambda_value, 3))

# Probability of a word appearing exactly k times
k = 2
poisson_prob = poisson.pmf(k, lambda_value)

print(f"Probability of a word appearing exactly {k} times (Poisson):", round(poisson_prob, 4))


# -----------------------------------
# STEP 5 – Binomial Distribution Modeling
# -----------------------------------

# Pick a specific word
target_word = "language"
word_count = freq_dist[target_word]

# Probability of that word in one trial
p = word_probabilities[target_word]

# Model as binomial over total words
binomial_prob = binom.pmf(word_count, total_words, p)

print("\n===== BINOMIAL MODEL =====")
print(f"Word '{target_word}' count:", word_count)
print("Binomial Probability:", round(binomial_prob, 6))


# -----------------------------------
# STEP 6 – Interpretation
# -----------------------------------

print("\n===== INTERPRETATION =====")
print("1. Word probability = frequency / total words.")
print("2. Poisson models rare independent events across vocabulary.")
print("3. Binomial models specific word occurrence across trials.")
print("4. Common words have higher probability mass.")
print("5. Rare words tend toward low-probability tail behavior.")
print("6. Larger datasets smooth probability distributions.")

