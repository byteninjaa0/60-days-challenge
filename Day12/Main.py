import re
import nltk
from typing import List, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class NLPPipeline:
    def __init__(
        self,
        remove_stopwords: bool = True,
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        vectorizer_type: str = None
    ):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.vectorizer_type = vectorizer_type

        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.vectorizer = None

    # ----------------------------
    # Step 1: Cleaning
    # ----------------------------
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ----------------------------
    # Step 2: Tokenization
    # ----------------------------
    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    # ----------------------------
    # Step 3: Stopword Removal
    # ----------------------------
    def remove_stopword_tokens(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    # ----------------------------
    # Step 4: Stemming / Lemmatization
    # ----------------------------
    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        if self.use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]

        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return tokens

    # ----------------------------
    # Full Preprocessing Pipeline
    # ----------------------------
    def preprocess(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]

        processed_texts = []

        for text in texts:
            text = self.clean_text(text)
            tokens = self.tokenize(text)

            if self.remove_stopwords:
                tokens = self.remove_stopword_tokens(tokens)

            tokens = self.normalize_tokens(tokens)

            processed_texts.append(" ".join(tokens))

        return processed_texts

    # ----------------------------
    # Optional Vectorization
    # ----------------------------
    def vectorize(self, texts: List[str]):
        if self.vectorizer_type == "bow":
            self.vectorizer = CountVectorizer()
        elif self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer()
        else:
            return None

        vectors = self.vectorizer.fit_transform(texts)
        return vectors.toarray()


# -----------------------------------------
# Example Usage
# -----------------------------------------
if __name__ == "__main__":

    documents = [
        "Machine Learning is transforming the world!",
        "NLP pipelines make machine learning scalable.",
        "Reusable code improves engineering quality."
    ]

    pipeline = NLPPipeline(
        remove_stopwords=True,
        use_stemming=False,
        use_lemmatization=True,
        vectorizer_type="tfidf"
    )

    processed_docs = pipeline.preprocess(documents)

    print("Processed Text:")
    for doc in processed_docs:
        print(doc)

    vectors = pipeline.vectorize(processed_docs)

    if vectors is not None:
        print("\nVector Representation:")
        print(vectors)
