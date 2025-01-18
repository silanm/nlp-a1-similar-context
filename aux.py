import nltk
import numpy as np
import torch
import torch.nn as nn

nltk.download("reuters")
nltk.download("punkt")

from collections import Counter

from nltk.corpus import reuters
from scipy.stats import spearmanr

# SAMPLE_SIZE = 1000
SAMPLE_SIZE = len(reuters.fileids())


def build_corpus(sample_size):
    corpus = []
    for file_id in reuters.fileids()[:sample_size]:
        words = [word.lower() for word in reuters.words(file_id) if word.isalpha()]
        corpus.append(words)
    return corpus


def build_vocab(corpus, min_freq=5):
    words = [word for sentence in corpus for word in sentence]
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    vocab.append("<UNKNOWN>")
    word2index = {word: idx for idx, word in enumerate(vocab)}
    word2index["<UNKNOWN>"] = 0
    return vocab, word2index, word_counts


def build_word2index(vocab):
    return {w: i for i, w in enumerate(vocab)}


def build_index2word(word2index):
    return {v: k for k, v in word2index.items()}


def get_embedding(model, word_index):
    return model.embedding_v(torch.tensor([word_index], dtype=torch.long).to("cpu")).detach().cpu().numpy()


def compare_similarity(model, word1, word2, similarity_file, word2index):
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


# Skipgram Model
class Skipgram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Skipgram, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)

    def forward(self, center_words, context_words):
        # Get the embeddings for the center and context words
        center_embed = self.embedding_v(center_words)
        context_embed = self.embedding_u(context_words)

        # Compute the scores by taking the dot product of center and context embeddings
        scores = torch.matmul(center_embed, context_embed.T)

        # Apply log softmax to the scores to get the log probabilities
        log_probs = torch.log_softmax(scores, dim=1)
        return log_probs


# Skipgram Model with Negative Sampling
class SkipgramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, center_words, pos_context, neg_context):
        # Get the embeddings for the center, positive context, and negative context words
        center_embed = self.embedding_v(center_words)
        pos_embed = self.embedding_u(pos_context)
        neg_embed = self.embedding_u(neg_context)

        # Compute the positive score by taking the dot product of center and positive context embeddings
        pos_score = self.logsigmoid(torch.bmm(pos_embed.unsqueeze(1), center_embed.unsqueeze(2))).squeeze()

        # Compute the negative score by taking the dot product of center and negative context embeddings
        neg_score = self.logsigmoid(-torch.bmm(neg_embed, center_embed.unsqueeze(2))).squeeze()

        # Calculate the loss as the negative sum of positive and negative scores
        loss = -(pos_score.sum() + neg_score.sum())
        return loss


# GloVe Model
class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(GloVe, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)
        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, co_occurrences, weightings):
        # Get the embeddings for the center and target words
        center_embed = self.embedding_v(center_words)
        target_embed = self.embedding_u(target_words)

        # Get the biases for the center and target words
        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        # Compute the inner product of the center and target embeddings
        inner_product = (center_embed * target_embed).sum(dim=1)

        # Calculate the loss using the weighting function and co-occurrences
        loss = weightings * torch.pow(inner_product + center_bias + target_bias - co_occurrences, 2)
        return loss.mean()


def load_model(filepath, model_class, vocab_size, embed_size):
    checkpoint = torch.load(filepath, weights_only=False)
    model = model_class(vocab_size, embed_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["word2index"], checkpoint["vocab"]
