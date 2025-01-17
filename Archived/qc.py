# Skipgram Model with and without Negative Sampling using Reuters Dataset

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import nltk
from nltk.corpus import reuters

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Download necessary NLTK datasets
nltk.download("reuters")
nltk.download("punkt")


# Build Reuters Corpus
def build_corpus(sample_size=200):
    corpus = []
    for file_id in reuters.fileids()[:sample_size]:
        words = [word.lower() for word in reuters.words(file_id) if word.isalpha()]
        corpus.append(words)
    return corpus


corpus = build_corpus()


# Vocabulary Building
def build_vocab(corpus):
    words = [word for sentence in corpus for word in sentence]
    vocab = list(set(words))
    word2index = {word: idx for idx, word in enumerate(vocab)}
    index2word = {idx: word for word, idx in word2index.items()}
    return vocab, word2index, index2word


vocab, word2index, index2word = build_vocab(corpus)


# Generate Skip-grams
def build_skipgrams(corpus, word2index, window_size=1):
    skip_grams = []
    for sentence in corpus:
        for idx, word in enumerate(sentence):
            center = word2index[word]
            context_window = sentence[max(0, idx - window_size) : idx] + sentence[idx + 1 : idx + window_size + 1]
            for context_word in context_window:
                skip_grams.append((center, word2index[context_word]))
    return skip_grams


skip_grams = build_skipgrams(corpus, word2index)


# Skipgram Model without Negative Sampling
class Skipgram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Skipgram, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)

    def forward(self, center_words, context_words):
        center_embed = self.embedding_v(center_words)
        context_embed = self.embedding_u(context_words)
        scores = torch.matmul(center_embed, context_embed.T)
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
        center_embed = self.embedding_v(center_words)
        pos_embed = self.embedding_u(pos_context)
        neg_embed = self.embedding_u(neg_context)
        pos_score = self.logsigmoid(torch.bmm(pos_embed.unsqueeze(1), center_embed.unsqueeze(2))).squeeze()
        neg_score = self.logsigmoid(-torch.bmm(neg_embed, center_embed.unsqueeze(2))).squeeze()
        loss = -(pos_score.sum() + neg_score.sum())
        return loss


# Negative Sampling
def negative_sampling(word_counts, word2index, num_samples):
    total_count = sum(word_counts.values())
    word_freqs = np.array([word_counts[word] / total_count for word in word2index])
    unigram_dist = word_freqs**0.75
    unigram_dist /= unigram_dist.sum()
    return np.random.choice(len(word2index), size=num_samples, p=unigram_dist)


# Training Function
def train_skipgram(model, skip_grams, word_counts, word2index, epochs, batch_size, learning_rate, num_neg_samples=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(skip_grams)
        for i in range(0, len(skip_grams), batch_size):
            batch = skip_grams[i : i + batch_size]
            center_words, context_words = zip(*batch)
            center_words = torch.LongTensor(center_words).to(device)
            context_words = torch.LongTensor(context_words).to(device)

            optimizer.zero_grad()

            if isinstance(model, Skipgram):
                log_probs = model(center_words, context_words)
                loss = -torch.mean(log_probs[range(batch_size), context_words])
            else:
                neg_samples = torch.LongTensor([negative_sampling(word_counts, word2index, num_neg_samples) for _ in range(batch_size)]).to(device)
                loss = model(center_words, context_words, neg_samples)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# Main Execution
word_counts = Counter([word for sentence in corpus for word in sentence])
vocab_size = len(vocab)
embed_size = 50
epochs = 50
batch_size = 16
learning_rate = 0.001

# Train Skipgram without Negative Sampling
model_skipgram = Skipgram(vocab_size, embed_size).to(device)
train_skipgram(model_skipgram, skip_grams, word_counts, word2index, epochs, batch_size, learning_rate)

# Train Skipgram with Negative Sampling
model_skipgram_neg = SkipgramNegSampling(vocab_size, embed_size).to(device)
train_skipgram(model_skipgram_neg, skip_grams, word_counts, word2index, epochs, batch_size, learning_rate)

# Save Models
torch.save(model_skipgram.state_dict(), "skipgram_model.pth")
torch.save(model_skipgram_neg.state_dict(), "skipgram_neg_model.pth")
print("Models saved to disk.")


import random
import time
import math
from collections import Counter
from itertools import combinations_with_replacement

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import reuters
from scipy.stats import spearmanr

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Download necessary NLTK datasets
nltk.download("reuters")
nltk.download("punkt")

# Data Preparation
sample_size = 200
window_size = 2
min_word_freq = 5  # Frequency threshold for vocabulary trimming


def build_corpus():
    corpus = []
    for file_id in reuters.fileids()[:sample_size]:
        sentences = reuters.words(file_id)
        sentences = [word.lower() for word in sentences if word.isalpha()]
        corpus.append(sentences)
    return corpus


def build_vocab(corpus):
    words = [word for sentence in corpus for word in sentence]
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_word_freq]
    vocab.append("<UNKNOWN>")
    word2index = {word: idx for idx, word in enumerate(vocab)}
    word2index["<UNKNOWN>"] = 0
    return vocab, len(vocab), word2index, word_counts


def build_skipgrams(corpus, word2index, window_size):
    skip_grams = []
    for sentence in corpus:
        for pos, center_word in enumerate(sentence):
            center_idx = word2index.get(center_word, word2index["<UNKNOWN>"])
            context_indices = [
                word2index.get(sentence[i], word2index["<UNKNOWN>"])
                for i in range(max(pos - window_size, 0), min(pos + window_size + 1, len(sentence)))
                if i != pos
            ]
            for context_idx in context_indices:
                skip_grams.append((center_idx, context_idx))
    return skip_grams


def weighting_function(x_ij, x_max=100, alpha=0.75):
    return (x_ij / x_max) ** alpha if x_ij < x_max else 1


# GloVe Model
class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(GloVe, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)
        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, co_occurrences, weightings):
        center_embed = self.embedding_v(center_words)
        target_embed = self.embedding_u(target_words)
        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)
        inner_product = (center_embed * target_embed).sum(dim=1)
        loss = weightings * torch.pow(inner_product + center_bias + target_bias - co_occurrences, 2)
        return loss.mean()


def prepare_training_data(skip_grams, co_occurrence_matrix, word2index):
    training_data = []
    for center, context in skip_grams:
        co_occurrence = co_occurrence_matrix.get((center, context), 1)
        weight = weighting_function(co_occurrence)
        training_data.append((center, context, math.log(co_occurrence + 1), weight))
    return training_data


# Training Function
def train_glove_model(model, training_data, epochs, batch_size, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i : i + batch_size]
            centers, contexts, coocs, weights = zip(*batch)
            centers = torch.LongTensor(centers).to(device)
            contexts = torch.LongTensor(contexts).to(device)
            coocs = torch.FloatTensor(coocs).to(device)
            weights = torch.FloatTensor(weights).to(device)
            optimizer.zero_grad()
            loss = model(centers, contexts, coocs, weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# Main Execution
corpus = build_corpus()
vocab, vocab_size, word2index, word_counts = build_vocab(corpus)
skip_grams = build_skipgrams(corpus, word2index, window_size)
co_occurrence_matrix = Counter(skip_grams)
training_data = prepare_training_data(skip_grams, co_occurrence_matrix, word2index)

embedding_dim = 100
epochs = 50
batch_size = 64
learning_rate = 0.01

model = GloVe(vocab_size, embedding_dim).to(device)
train_glove_model(model, training_data, epochs, batch_size, learning_rate)

# Save the model
torch.save({"model_state_dict": model.state_dict(), "word2index": word2index, "vocab": vocab}, "glove_model.pth")
print("Model and vocabulary saved to glove_model.pth")
