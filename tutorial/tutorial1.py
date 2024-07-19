import nltk
import random
from collections import defaultdict

# Download necessary NLTK data
nltk.download('punkt')

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    # Tokenize the text into sentences and words
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    return words