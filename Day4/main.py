import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
text = input("Enter noisy text:\n")
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ''.join([char for char in text if not char.isdigit()])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)
cleaned_text = clean_text(text)

print("\nCleaned Text:")
print(cleaned_text)