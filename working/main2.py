import nltk
import random
from collections import defaultdict

# Download necessary NLTK data
nltk.download("punkt")


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def preprocess_text(text):
    # Tokenize the text into sentences and words
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    return words


def build_markov_chain(words, n=2):
    markov_chain = defaultdict(list)
    for sentence in words:
        if len(sentence) > n:
            for i in range(len(sentence) - n):
                state = tuple(sentence[i : i + n])
                next_word = sentence[i + n]
                markov_chain[state].append(next_word)
    return markov_chain


def generate_sentence(markov_chain, n=2, max_length=20):
    if not markov_chain:
        return "Not enough data to generate a sentence."
    current = random.choice(list(markov_chain.keys()))
    result = list(current)
    for _ in range(max_length - n):
        if current not in markov_chain:
            break
        next_word = random.choice(markov_chain[current])
        result.append(next_word)
        current = tuple(result[-n:])
        if next_word.endswith("."):
            break
    return " ".join(result)


def generate_chapter(markov_chain, num_sentences=10):
    return " ".join(generate_sentence(markov_chain) for _ in range(num_sentences))


# Main execution
if __name__ == "__main__":
    # Load and preprocess the text
    text = load_text("training_data.txt")
    print("Length of loaded text:", len(text))
    processed_text = preprocess_text(text)
    print("Number of sentences:", len(processed_text))
    print("First few words:", processed_text[0][:10] if processed_text else "No words")

    # Build the Markov chain
    markov_chain = build_markov_chain(processed_text)
    print("Size of Markov chain:", len(markov_chain))

    # Only generate if we have data
    if markov_chain:
        # Generate a chapter
        chapter = generate_chapter(markov_chain)
        print(chapter)
    else:
        print("Not enough data to generate text")
