# Unified Model: Skipgram, Negative Sampling, and GloVe using Reuters Dataset

import random
import time
import math
from collections import Counter
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import reuters

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Download necessary NLTK datasets
# nltk.download("reuters")
# nltk.download("punkt")


# Build Reuters Corpus
def build_corpus(sample_size=100):
    corpus = []
    for file_id in reuters.fileids()[:sample_size]:
        words = [word.lower() for word in reuters.words(file_id) if word.isalpha()]
        corpus.append(words)
    return corpus


# Vocabulary Building
def build_vocab(corpus, min_freq=5):
    words = [word for sentence in corpus for word in sentence]
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    vocab.append("<UNKNOWN>")
    word2index = {word: idx for idx, word in enumerate(vocab)}
    word2index["<UNKNOWN>"] = 0
    return vocab, word2index, word_counts


# Generate Skip-grams
def build_skipgrams(corpus, word2index, window_size=2):
    skip_grams = []
    for sentence in corpus:
        for idx, word in enumerate(sentence):
            center = word2index.get(word, word2index["<UNKNOWN>"])
            context_window = sentence[max(0, idx - window_size) : idx] + sentence[idx + 1 : idx + window_size + 1]
            for context_word in context_window:
                context = word2index.get(context_word, word2index["<UNKNOWN>"])
                skip_grams.append((center, context))
    return skip_grams


def weighting_function(x_ij, x_max=100, alpha=0.75):
    return (x_ij / x_max) ** alpha if x_ij < x_max else 1


# Skipgram Model
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


# Plotting Loss Function
def plot_losses(losses, model_name):
    plt.plot(losses)
    plt.title(f"Training Loss - {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


# Training Function for Skipgram Models
def train_skipgram(model, skip_grams, epochs, batch_size, learning_rate, word2index, num_neg_samples=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        random.shuffle(skip_grams)
        for i in range(0, len(skip_grams), batch_size):
            batch = skip_grams[i : i + batch_size]
            center_words, context_words = zip(*batch)
            center_words = torch.LongTensor(center_words).to(device)
            context_words = torch.LongTensor(context_words).to(device)

            if len(batch) < batch_size:
                padding_size = batch_size - len(batch)
                center_words = torch.cat([center_words, torch.zeros(padding_size, dtype=torch.long).to(device)])
                context_words = torch.cat([context_words, torch.zeros(padding_size, dtype=torch.long).to(device)])

            optimizer.zero_grad()
            if isinstance(model, Skipgram):
                log_probs = model(center_words, context_words)
                loss = -torch.mean(log_probs[range(batch_size), context_words])
            else:
                neg_samples = torch.LongTensor([np.random.choice(len(word2index), num_neg_samples) for _ in range(batch_size)]).to(device)
                loss = model(center_words, context_words, neg_samples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Model: {model.__class__.__name__:20s}| Epoch: {epoch + 1:-3d}/{epochs}  Loss: {total_loss:12.4f}  Time: {time.time() - start_time:6.2f}s"
        )
        loss_history.append(total_loss)
    plot_losses(loss_history, model.__class__.__name__)


def train_glove_model(model, training_data, epochs, batch_size, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
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
        print(
            f"Model: {model.__class__.__name__:20s}| Epoch: {epoch + 1:-3d}/{epochs}  Loss: {total_loss:12.4f}  Time: {time.time() - start_time:6.2f}s"
        )
        loss_history.append(total_loss)
    plot_losses(loss_history, model.__class__.__name__)


# Train Models
embedding_dim = 100
epochs = 10
batch_size = 64
learning_rate = 0.001

corpus = build_corpus()
vocab, word2index, word_counts = build_vocab(corpus)
skip_grams = build_skipgrams(corpus, word2index)


print("----------------------------------------------------------------------------")
model_skipgram = Skipgram(len(vocab), embedding_dim).to(device)
train_skipgram(model_skipgram, skip_grams, epochs, batch_size, learning_rate, word2index)
print("----------------------------------------------------------------------------")
model_skipgram_neg = SkipgramNegSampling(len(vocab), embedding_dim).to(device)
train_skipgram(model_skipgram_neg, skip_grams, epochs, batch_size, learning_rate, word2index)
print("----------------------------------------------------------------------------")
model_glove = GloVe(len(vocab), embedding_dim).to(device)
co_occurrence_matrix = Counter(skip_grams)  # Prepare co-occurrence matrix
training_data_glove = [(center, context, math.log(count + 1), weighting_function(count)) for (center, context), count in co_occurrence_matrix.items()]
train_glove_model(model_glove, training_data_glove, epochs, batch_size, learning_rate)
print("----------------------------------------------------------------------------")

# Save Models
torch.save({"model_state_dict": model_skipgram.state_dict(), "word2index": word2index, "vocab": vocab}, "skipgram_model.pth")
torch.save({"model_state_dict": model_skipgram_neg.state_dict(), "word2index": word2index, "vocab": vocab}, "skipgram_neg_model.pth")
torch.save({"model_state_dict": model_glove.state_dict(), "word2index": word2index, "vocab": vocab}, "glove_model.pth")
print("All models have been saved.")


# Load Models


# Evaluate Analogies
# TODO: Find Semantic Accuracy score using capital-common-countries section of word-test.v1.txt
# TODO: Find Syntactic Accuracy score using gram7-past-tense section of word-test.v1.txt
def eval_analogies():
    pass


# Evaluate Similarity
# TODO: Find the correlation between your models’ dot product and the provided similarity metrics (spearmanr). Output is MSE score.
# TODO: Assess if your embeddings correlate with human judgment of word similarity against the wordsim_similarity_goldstandard.txt file
def eval_similarity():
    pass
