import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
def normalize_case(text: str) -> str:
    return text.lower()


def remove_urls_mentions_hashtags(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    return text


def remove_punctuation_numbers(text: str) -> str:
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
def tokenize(text: str) -> List[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _get_wordnet_pos(treebank_tag: str):
    """Map NLTK POS tag to WordNet POS tag."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def lemmatize(tokens: List[str]) -> List[str]:
    tagged = nltk.pos_tag(tokens)
    return [
        LEMMATIZER.lemmatize(token, _get_wordnet_pos(pos))
        for token, pos in tagged
    ]
def preprocess_text(text: Optional[str]) -> str:
    if not text or not text.strip():
        return ""

    text = normalize_case(text)
    text = remove_urls_mentions_hashtags(text)
    text = remove_punctuation_numbers(text)
    text = normalize_whitespace(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)
def preprocess_corpus(texts: List[str]) -> List[str]:
    return [preprocess_text(text) for text in texts]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text to preprocess: ").strip()

    result = preprocess_text(text)
    print(result)
