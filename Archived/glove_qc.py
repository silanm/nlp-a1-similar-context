import random
import time
from collections import Counter

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import reuters
from scipy.stats import spearmanr

# Detect the device for computation (CPU/GPU/Metal on Mac 💻)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")


# Ensure required datasets are downloaded
nltk.download("reuters")
nltk.download("punkt")


sample_size = 200


def build_corpus():
    corpus = []

    for id in reuters.fileids()[:sample_size]:
        sentences = reuters.words(id)
        sentences = [sentence.lower() for sentence in sentences if sentence.isalpha()]
        corpus.append(sentences)

    return corpus


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


def build_skipgrams(corpus, word2index, window_size=2):
    """
    Generate skip-gram pairs from corpus
    """
    skip_grams = []
    skip_grams_words = []

    for sentence in corpus:
        for position, center_word in enumerate(sentence):
            center_index = word2index[center_word]
            context_indices = list(
                [
                    i
                    for i in range(
                        max(position - window_size, 0),  # Context words on the left. If none, then 0
                        min(position + window_size + 1, len(sentence)),  # Context words on the right, If none, then 0
                    )
                    if i != position  # Exclude itself
                ]
            )
            for index in context_indices:
                context_word = sentence[index]
                context_index = word2index[context_word]
                skip_grams.append((center_index, context_index))  # A tuple representing a skip-gram pair (indices)
                skip_grams_words.append((center_word, context_word))  # A tuple representing a skip-gram pair (words)

    return skip_grams, skip_grams_words


def to_number_sequence(all_vocabs, word2index):
    """
    Convert a sequence of words into a sequence of numerical indices
    """
    indices = list(
        map(  # Apply lambda function to each word in all_vocabs
            lambda w: (word2index[w] if word2index.get(w) is not None else word2index["<UNKNOWN>"]),
            all_vocabs,
        )
    )
    return torch.LongTensor(indices).to(device)  # List of indices is converted to PyTorch tensor


corpus = build_corpus()
all_vocabs, vocab_size, word2index, word_counts = build_vocab(corpus)
skip_grams, skip_grams_words = build_skipgrams(corpus, word2index, window_size=2)

X_ik_skipgram = Counter(skip_grams_words)
list(X_ik_skipgram.items())[:5]


def weighting(word_i, word_j, X_ik):
    try:
        x_ij = X_ik[(word_i, word_j)]
    except:
        x_ij = 1

    x_max = 100
    alpha = 0.75

    if x_ij < x_max:
        result = (x_ij / x_max) ** alpha
    else:
        result = 1

    return result


from itertools import combinations_with_replacement

# Initialize dictionaries
X_ik = {}
weighting_dict = {}

# Iterate over all possible bi-grams
for bi_gram in combinations_with_replacement(all_vocabs, 2):
    # Check if the bi-gram exists in the skip-gram counter
    co_occurrence = X_ik_skipgram.get(bi_gram)
    if co_occurrence is not None:
        # Update co-occurrence counts for both directions
        X_ik[bi_gram] = co_occurrence + 1
        X_ik[(bi_gram[1], bi_gram[0])] = co_occurrence + 1

        # Update weighting for both directions
        weighting_dict[bi_gram] = weighting(bi_gram[0], bi_gram[1], X_ik)
        weighting_dict[(bi_gram[1], bi_gram[0])] = weighting(bi_gram[1], bi_gram[0], X_ik)


class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(GloVe, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)
        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, co_occurrences, weighting):
        center_embeds = self.embedding_v(center_words)
        target_embeds = self.embedding_u(target_words)

        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)

        loss = weighting * torch.pow((inner_product + center_bias + target_bias), 2)

        return torch.sum(loss)


import math


def random_batch(batch_size, word_sequence, skip_grams, X_ik, weighting_dic):
    # print(skip_grams)
    # convert to id since our skip_grams is word, not yet id
    skip_grams_id = [(word2index[skip_gram[0]], word2index[skip_gram[1]]) for skip_gram in skip_grams]

    random_inputs = []
    random_labels = []
    random_coocs = []
    random_weightings = []
    random_index = np.random.choice(range(len(skip_grams_id)), batch_size, replace=False)  # randomly pick without replacement

    for i in random_index:
        random_inputs.append([skip_grams_id[i][0]])  # target, e.g., 2
        random_labels.append([skip_grams_id[i][1]])  # context word, e.g., 3

        # get cooc
        pair = skip_grams[i]
        try:
            cooc = X_ik[pair]
        except:
            cooc = 1
        random_coocs.append([math.log(cooc)])

        # get weighting
        weighting = weighting_dic[pair]
        random_weightings.append([weighting])

    return (
        np.array(random_inputs),
        np.array(random_labels),
        np.array(random_coocs),
        np.array(random_weightings),
    )


# Hyperparameters
batch_size = 64
embed_size = 100
epochs = 100
epoch_losses = []


def train_glove(skip_grams):
    for epoch in range(epochs):
        start_time = time.time()

        input_batch, target_batch, cooc_batch, weighting_batch = random_batch(batch_size, corpus, skip_grams, X_ik, weighting_dict)
        input_batch = torch.LongTensor(input_batch)  # [batch_size, 1]
        target_batch = torch.LongTensor(target_batch)  # [batch_size, 1]
        cooc_batch = torch.FloatTensor(cooc_batch)  # [batch_size, 1]
        weighting_batch = torch.FloatTensor(weighting_batch)  # [batch_size, 1]

        model = GloVe(vocab_size=vocab_size, embed_size=embed_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        optimizer.zero_grad()
        loss = model(input_batch, target_batch, cooc_batch, weighting_batch)

        # Save epoch and loss for later plots and analysis
        if epoch == 0:
            epoch_losses = []
        epoch_losses.append((epoch + 1, loss.item()))

        loss.backward()
        optimizer.step()

        elapsed_time = time.time() - start_time

        print(f"Epoch: {epoch + 1:-3d}/{epochs} | Loss: {loss:12.5f} | Time: {elapsed_time:12.5f}")
    return model, epoch_losses


model_glove, epoch_losses = train_glove(skip_grams=skip_grams_words)

model_glove_path = "glove.model"
torch.save(model_glove, model_glove_path)
print("Model saved to", model_glove_path)
