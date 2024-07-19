import nltk
import random
from collections import defaultdict

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    pos_tags = [nltk.pos_tag(sentence) for sentence in words]
    return pos_tags

def build_markov_chain(pos_tags, n=2):
    markov_chain = defaultdict(list)
    for sentence in pos_tags:
        for i in range(len(sentence) - n):
            state = tuple(sentence[i:i+n])
            next_word = sentence[i+n]
            markov_chain[state].append(next_word)
    return markov_chain

def generate_sentence(markov_chain, n=2, max_length=20):
    current = random.choice(list(markov_chain.keys()))
    result = list(current)
    for _ in range(max_length - n):
        if current not in markov_chain:
            break
        next_word = random.choice(markov_chain[current])
        result.append(next_word)
        current = tuple(result[-n:])
        if next_word[0].endswith('.'):
            break
    return ' '.join(word for word, tag in result)

def generate_chapter(markov_chain, num_sentences=10):
    return ' '.join(generate_sentence(markov_chain) for _ in range(num_sentences))

def add_structure(chapter):
    # Simple structure: beginning, middle, end
    structured_chapter = "Chapter X: A New Adventure\n\n"
    sentences = nltk.sent_tokenize(chapter)
    
    # Beginning
    structured_chapter += "In the beginning, " + sentences[0].lower() + " "
    structured_chapter += " ".join(sentences[1:3]) + "\n\n"
    
    # Middle
    structured_chapter += "As the story unfolded, " + " ".join(sentences[3:-2]) + "\n\n"
    
    # End
    structured_chapter += "Finally, " + " ".join(sentences[-2:])
    
    return structured_chapter

if __name__ == "__main__":
    text = load_text("training_data.txt")
    processed_text = preprocess_text(text)
    markov_chain = build_markov_chain(processed_text, n=3)
    chapter = generate_chapter(markov_chain, num_sentences=15)
    structured_chapter = add_structure(chapter)
    print(structured_chapter)