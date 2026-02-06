import re
from collections import Counter
import matplotlib.pyplot as plt

def preprocess_text(text, remove_stopwords=True):
    text = text.lower()
    words = re.findall(r"\b[a-z]+\b", text)

    if remove_stopwords:
        stopwords = {
            "the","is","in","and","to","of","a","an","on","for",
            "with","as","by","at","from","or","that","this","it",
            "are","was","were","be","has","have","had"
        }
        words = [word for word in words if word not in stopwords]

    return words


def word_frequency_distribution(text):
    tokens = preprocess_text(text)
    return Counter(tokens)


def plot_top_20_words(freq_dist):
    top_20 = freq_dist.most_common(20)

    words = [word for word, _ in top_20]
    counts = [count for _, count in top_20]

    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Top 20 Most Frequent Words")
    plt.tight_layout()
    plt.show()


# ------------------ User Input ------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text_corpus = " ".join(sys.argv[1:])
    else:
        print("Enter your text (press Enter twice to finish):")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                break
            lines.append(line)
        text_corpus = " ".join(lines)

    if not text_corpus.strip():
        print("No text entered. Exiting.")
        sys.exit(1)

    freq_distribution = word_frequency_distribution(text_corpus)

    print("\nWord Frequency Distribution:")
    for word, count in freq_distribution.items():
        print(f"{word}: {count}")

    plot_top_20_words(freq_distribution)
