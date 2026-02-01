import re
from collections import Counter

def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

def freq(arr):
    return Counter(arr)

def remove_stopwords(tokens):
    stopwords = {
        "a","an","the","and","or","but","is","are","was","were",
        "in","on","at","to","for","of","with","by","as","from"
    }
    return [word for word in tokens if word not in stopwords]

text = input().strip()

tokens = tokenize(text)
filtered_tokens = remove_stopwords(tokens)
frequency = freq(filtered_tokens)

print(tokens)
print(filtered_tokens)
print(frequency)
