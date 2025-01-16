import numpy as np
import nltk
import torch
from collections import Counter


nltk.download("reuters")
nltk.download("punkt")

from nltk.corpus import reuters
from scipy.stats import spearmanr

# SAMPLE_SIZE = 1000
SAMPLE_SIZE = len(reuters.fileids())


def build_corpus():
    corpus = []

    for id in reuters.fileids()[:SAMPLE_SIZE]:
        sentences = reuters.words(id)
        sentences = [
            sentence.lower() for sentence in sentences if sentence.isalpha()
        ]
        corpus.append(sentences)

    return corpus


def build_vocab(corpus):
    flatten = lambda l: [item for sublist in l for item in sublist]
    vocab = list(set(flatten(corpus)))
    return vocab, len(vocab)


def build_word2index(vocab):
    return {w: i for i, w in enumerate(vocab)}


def build_index2word(word2index):
    return {v: k for k, v in word2index.items()}


def get_embedding(model, word_index):
    return (
        model.embedding_v(
            torch.tensor([word_index], dtype=torch.long).to('cpu')
        ).detach().cpu().numpy()
    )


def compare_similarity(model, word1, word2, similarity_file, word2index):
    """Compare similarity score of two words with human score."""
        
    # Get embeddings for both words
    word1_vec = get_embedding(model, word2index[word1]).squeeze()
    word2_vec = get_embedding(model, word2index[word2]).squeeze()

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

    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr([model_score], [human_score])
    print(spearman_corr)
    return model_score, human_score, spearman_corr


def compare_similarity_gensim(model, word1, word2, similarity_file, word2index):
    
    model_score = model.similarity(word1, word2)
    human_score = None
    with open(similarity_file, "r", encoding="utf-8") as f:
        for line in f:
            w1, w2, score = line.strip().split()
            if (w1 == word1 and w2 == word2) or (w1 == word2 and w2 == word1):
                human_score = float(score)
                break
            
    if human_score is None:
        return f"No human score found for words: {word1}, {word2}"
    
    spearman_corr, _ = spearmanr([model_score], [human_score])
    
    return model_score, human_score, spearman_corr