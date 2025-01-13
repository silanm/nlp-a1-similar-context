import gradio as gr
import numpy as np
import torch
import Skipgram
import nltk
from nltk.corpus import reuters
from collections import Counter

MODEL_SKIPGRAM_SM_PATH = "model_skipgram_sm_b128_em100_ep10.pth"
MODEL_SKIPGRAM_NG_PATH = "model_skipgram_ng_b128_em100_ep10_neg5.pth"
PATH_WORDSIM = "wordsim_similarity_goldstandard.txt"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
)

sample_size = 500
# sample_size = len(reuters.fileids())


def build_corpus():
    corpus = []

    for id in reuters.fileids()[:sample_size]:
        sentences = reuters.words(id)
        sentences = [sentence.lower() for sentence in sentences if sentence.isalpha()]
        corpus.append(sentences)

    return corpus


corpus = build_corpus()


def build_vocab(corpus):
    # Flatten words from all sentences
    flatten_words = [word for sentence in corpus for word in sentence]
    word_counts = Counter(flatten_words)

    # Unique words
    vocab = list(set(flatten_words))
    vocab.append("<UNKNOWN>")

    word2index = {word: index for index, word in enumerate(vocab)}
    word2index["<UNKNOWN>"] = 0
    word2index

    return vocab, len(vocab), word2index, word_counts


all_vocabs, vocab_size, word2index, word_counts = build_vocab(corpus)


loaded_model_sm = Skipgram.Skipgram(vocab_size=len(all_vocabs), embed_size=100, mode="softmax").to(
    device
)
loaded_model_sm.load_state_dict(
    torch.load(MODEL_SKIPGRAM_SM_PATH, weights_only=True, map_location=device)
)

loaded_model_ng = Skipgram.Skipgram(
    vocab_size=len(all_vocabs), embed_size=100, mode="negative_sampling"
).to(device)
loaded_model_ng.load_state_dict(
    torch.load(MODEL_SKIPGRAM_NG_PATH, weights_only=True, map_location=device)
)

print("Models loaded")


def compare_similarity(model, word1, word2, similarity_file, word2index):
    """Compare similarity score of two words with human score."""
    # Get embeddings for both words
    word1_vec = model.get_embedding(word2index[word1]).squeeze()
    word2_vec = model.get_embedding(word2index[word2]).squeeze()

    # Calculate the similarity score using dot product
    model_score = np.dot(word1_vec, word2_vec)

    # Read the human score from the similarity file
    human_score = None
    with open(similarity_file, "r", encoding="utf-8") as f:
        for line in f:
            w1, w2, score = line.strip().split()
            if (w1 == word1 and w2 == word2) or (w1 == word2 and w2 == word1):
                human_score = float(score)
                break

    if human_score is None:
        return f"No human score found for words: {word1}, {word2}"

    return model_score, human_score


# Dummy function to calculate similarity
def calculate_similarity(word1, word2):
    """
    Calculate the similarity between two words using a model and a gold standard.

    Args:
        word1 (str): The first word.
        word2 (str): The second word.

    Returns:
        tuple: A tuple containing two floats:
            - model_similarity (float): The similarity score calculated by the model.
            - gold_standard_similarity (float): The similarity score from the gold standard.
    """
    # Replace this with actual model similarity calculation
    model_similarity = np.random.rand()
    # Replace this with actual gold standard similarity lookup
    gold_standard_similarity = np.random.rand()
    return model_similarity, gold_standard_similarity


def similarity_interface(word1, word2):
    """
    Calculate and return the similarity between two words.

    This function takes two words as input, calculates their similarity using a model and a gold standard,
    and returns a list containing the words and their respective similarity scores.

    Args:
        word1 (str): The first word to compare.
        word2 (str): The second word to compare.

    Returns:
        list: A list containing a single sublist with the following elements:
            - word1 (str): The first word.
            - word2 (str): The second word.
            - model_similarity (float): The similarity score calculated by the model.
            - gold_standard_similarity (float): The similarity score according to the gold standard.
    """
    model_similarity, gold_standard_similarity = calculate_similarity(word1, word2)
    model_score, human_score = compare_similarity(
        loaded_model_sm,
        word1=word1,
        word2=word2,
        similarity_file=PATH_WORDSIM,
        word2index=word2index,
    )
    return [[word1, word2, model_score, human_score]]
    # return [[word1, word2, model_similarity, gold_standard_similarity]]


# Define the Gradio interface
with gr.Blocks() as demo:
    with gr.Tab("Find Similarities"):
        word1_input = gr.Textbox(label="Word 1")
        word2_input = gr.Textbox(label="Word 2")
        calculate_button = gr.Button("Calculate")
        result_table = gr.Dataframe(
            headers=["Word 1", "Word 2", "Model Similarity", "Gold Standard Similarity"]
        )

        calculate_button.click(
            fn=similarity_interface, inputs=[word1_input, word2_input], outputs=result_table
        )

    with gr.Tab("Blank Tab"):
        pass

# Launch the Gradio app
demo.launch()
