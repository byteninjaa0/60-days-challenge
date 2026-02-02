import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
text = input()
sentences = sent_tokenize(text)
print("Sentence Tokens:")
for s in sentences:
    print(s)

print("\nWord Tokens:")
for s in sentences:
    words = word_tokenize(s)
    print(words)
