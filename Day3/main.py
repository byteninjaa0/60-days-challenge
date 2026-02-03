import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens
stemmer = PorterStemmer()

def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]
lemmatizer = WordNetLemmatizer()

def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]
reviews = [
    "I loved the products! They are running smoothly and worked perfectly.",
    "The batteries were dying faster than expected.",
    "This phone has better cameras and improved features."
]

for review in reviews:
    cleaned = clean_text(review)
    stemmed = apply_stemming(cleaned)
    lemmatized = apply_lemmatization(cleaned)

    print("Original Review:", review)
    print("Cleaned Tokens:", cleaned)
    print("Stemmed:", stemmed)
    print("Lemmatized:", lemmatized)
    print("-" * 60)
