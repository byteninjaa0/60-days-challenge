import re
from typing import List, Union

try:
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    # Download NLTK data (silent=True to avoid prompts)
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Warning: NLTK data download issue: {e}")
        print("Attempting to continue anyway...")
    
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    
except ImportError as e:
    print(f"Error: Required library not installed: {e}")
    print("Please install required packages:")
    print("  pip install nltk scikit-learn")
    exit(1)


class SpamPreprocessingPipeline:

    def __init__(
        self,
        remove_stopwords: bool = True,
        keep_negations: bool = True,
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        vectorizer_type: str = "tfidf"
    ):
        self.remove_stopwords = remove_stopwords
        self.keep_negations = keep_negations
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.vectorizer_type = vectorizer_type

        self.stop_words = set(stopwords.words("english"))

        if self.keep_negations:
            self.stop_words = self.stop_words - {"not", "no", "nor"}

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None

    # -----------------------
    # Step 1: Cleaning
    # -----------------------
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)      
        text = re.sub(r"[^a-z\s]", "", text)     
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -----------------------
    # Step 2: Tokenization
    # -----------------------
    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    # -----------------------
    # Step 3: Stopword Removal
    # -----------------------
    def remove_stopword_tokens(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    # -----------------------
    # Step 4: Normalization
    # -----------------------
    def normalize(self, tokens: List[str]) -> List[str]:
        if self.use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]

        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return tokens

    # -----------------------
    # Full Pipeline
    # -----------------------
    def preprocess(self, texts: Union[str, List[str]]) -> List[str]:

        if isinstance(texts, str):
            texts = [texts]

        processed_texts = []

        for text in texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)

            if self.remove_stopwords:
                tokens = self.remove_stopword_tokens(tokens)

            tokens = self.normalize(tokens)

            processed_texts.append(" ".join(tokens))

        return processed_texts

    # -----------------------
    # Vectorization
    # -----------------------
    def vectorize(self, texts: List[str]):
        if self.vectorizer_type == "bow":
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = TfidfVectorizer()

        vectors = self.vectorizer.fit_transform(texts)
        return vectors.toarray()


# -------------------------------------------------
# Example Usage
# -------------------------------------------------
if __name__ == "__main__":

    messages = [
        "Congratulations!!! You won $1000. Click here NOW!!!",
        "Hey, are we meeting tomorrow?",
        "Free entry in a weekly contest. Text WIN to 12345",
        "Please call me when you reach home."
    ]

    pipeline = SpamPreprocessingPipeline(
        remove_stopwords=True,
        keep_negations=True,
        use_stemming=False,
        use_lemmatization=True,
        vectorizer_type="tfidf"
    )

    print("===== BEFORE PROCESSING =====")
    for msg in messages:
        print(msg)

    processed = pipeline.preprocess(messages)

    print("\n===== AFTER PROCESSING =====")
    for msg in processed:
        print(msg)

    vectors = pipeline.vectorize(processed)

    print("\n===== VECTOR REPRESENTATION =====")
    print(vectors)
    