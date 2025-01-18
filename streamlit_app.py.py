import random

import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import aux
import streamlit as st

from nltk.corpus import reuters

PATH_MODEL_W2V_SKIPGRAM = "model/skipgram_model.pth"
PATH_MODEL_W2V_NEGATIVE = "model/skipgram_neg_model.pth"
PATH_MODEL_GLV = "mode/glove_model.pth"
PATH_GLOVE_100D = "glove.6B.100d.txt"
PATH_WORDSIM = "wordsim_similarity_goldstandard.txt"
PATH_WORDTEST = "word-test.v1.txt"
SAMPLE_SIZE = 100  # Number of documents to sample from the Reuters corpus
SAMPLE_SIZE = len(reuters.fileids())  # Number of documents to sample from the Reuters corpus
EMBEDDING_DIMENSION = 100  # Dimension of the embedding vectors

device = "cpu"
corpus = aux.build_corpus(SAMPLE_SIZE)
vocab, word2index, word_counts = aux.build_vocab(corpus)


# Load models (assuming these are trained and available)
def load_model(filepath, model_class, vocab_size, embed_size):
    checkpoint = torch.load(filepath, weights_only=False, map_location=device)
    model = model_class(vocab_size, embed_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["word2index"], checkpoint["vocab"]


def load_gensim_glove(filepath):
    model = KeyedVectors.load_word2vec_format(filepath, binary=False, no_header=True)
    return model


# Load models
model_skipgram, word2index_skipgram, vocab_skipgram = load_model(PATH_MODEL_W2V_SKIPGRAM, aux.Skipgram, len(vocab), EMBEDDING_DIMENSION)
model_skipgram_neg, word2index_neg, vocab_neg = load_model(PATH_MODEL_W2V_NEGATIVE, aux.SkipgramNegSampling, len(vocab), EMBEDDING_DIMENSION)
model_glove, word2index_glove, vocab_glove = load_model(PATH_MODEL_GLV, aux.GloVe, len(vocab), EMBEDDING_DIMENSION)
gensim_glove_model = load_gensim_glove(datapath(PATH_GLOVE_100D))


# Function to find top 10 similar words
def find_top_similar_words(model, word2index, index2word, input_word, top_n=10, is_gensim=False):
    results = []
    if is_gensim:
        if input_word not in model:
            return ["Input word not in vocabulary."]
        similar_words = model.most_similar(input_word, topn=top_n)
        results = [f"{word}: {similarity:.4f}" for word, similarity in similar_words]
    else:
        if input_word not in word2index:
            return ["Input word not in vocabulary."]
        input_vec = model.embedding_v(torch.tensor([word2index[input_word]])).detach()
        similarities = torch.matmul(model.embedding_v.weight, input_vec.squeeze())
        probabilities = torch.softmax(similarities, dim=0)
        top_indices = torch.topk(probabilities, top_n + 1).indices.tolist()[1:]
        results = [f"{index2word[idx]:10s}: {probabilities[idx]:15.4f}" for idx in top_indices]
    return results


# Streamlit UI
st.title("Similar Words Finder")

# Input section
col1, col2 = st.columns([2, 1])
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if st.button("Randomize Word"):
    st.session_state.user_input = random.choice(vocab)
    # st.experimental_rerun()

with col1:
    user_input = st.text_input(
        "Enter a word:",
        key="user_input",
        value=st.session_state.user_input,
    )

# Results section
if user_input:
    st.subheader("Top 10 Similar Words:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("**Skipgram**")
        skipgram_results = find_top_similar_words(model_skipgram, word2index_skipgram, vocab_skipgram, user_input)
        st.write("\n".join(skipgram_results))

    with col2:
        st.write("**Negative Sampling**")
        skipgram_neg_results = find_top_similar_words(model_skipgram_neg, word2index_neg, vocab_neg, user_input)
        st.write("\n".join(skipgram_neg_results))

    with col3:
        st.write("**GloVe**")
        glove_results = find_top_similar_words(model_glove, word2index_glove, vocab_glove, user_input)
        st.write("\n".join(glove_results))

    with col4:
        st.write("**GloVe Gensim**")
        gensim_results = find_top_similar_words(gensim_glove_model, gensim_glove_model.key_to_index, None, user_input, is_gensim=True)
        st.write("\n".join(gensim_results))
