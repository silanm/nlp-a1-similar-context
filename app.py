import gradio as gr
import numpy as np
import torch

from ChakyGloVe import GloVe
from ChakySkipgram import Skipgram
from ChakySkipgramNeg import SkipgramNegSampling

from nltk.corpus import reuters

import aux

PATH_MODEL_W2V_SKIPGRAM = "skipgram.model"
PATH_MODEL_W2V_NEGATIVE = "skipgram_neg_sampling.model"
PATH_MODEL_GLV = "glove.model"
PATH_MODEL_GLV_GENSIM = "glove_gensim.model"
PATH_WORDSIM = "wordsim_similarity_goldstandard.txt"
PATH_WORDTEST = "word-test.v1.txt"

# device = torch.device(
#     "mps"
#     if torch.backends.mps.is_available()
#     else ("cuda" if torch.cuda.is_available() else "cpu")
# )
device = "cpu"

corpus = aux.build_corpus()
vocab, vocab_size = aux.build_vocab(corpus)
word2index = aux.build_word2index(vocab)
index2word = aux.build_index2word(word2index)

loaded_model_w2v_skipgram = torch.load(
    PATH_MODEL_W2V_SKIPGRAM, weights_only=False, map_location=device
)

loaded_model_w2v_negative = torch.load(
    PATH_MODEL_W2V_NEGATIVE, weights_only=False, map_location=device
)

loaded_model_glv = torch.load(PATH_MODEL_GLV, weights_only=False, map_location=device)

loaded_model_glv_gensim = torch.load(PATH_MODEL_GLV_GENSIM, weights_only=False, map_location=device)


loaded_model_w2v_skipgram.eval()
loaded_model_w2v_negative.eval()
# loaded_model_glv.eval()

print("All models loaded")


def event_calc_similarity(word1, word2):
    model_w2v_skipgram_score, human_score, w2v_skipgram_spearman_corr = aux.compare_similarity(
        loaded_model_w2v_skipgram,
        word1=word1,
        word2=word2,
        similarity_file=PATH_WORDSIM,
        word2index=word2index,
    )
    model_w2v_negative_score, human_score, w2v_negative_spearman_corr = aux.compare_similarity(
        loaded_model_w2v_negative,
        word1=word1,
        word2=word2,
        similarity_file=PATH_WORDSIM,
        word2index=word2index,
    )
    model_glv_score, human_score, glove_spearman_corr = aux.compare_similarity_gensim(
        loaded_model_glv,
        word1=word1,
        word2=word2,
        similarity_file=PATH_WORDSIM,
        word2index=word2index,
    )
    model_glv_gensim_score, human_score, gensim_spearman_corr = aux.compare_similarity_gensim(
        loaded_model_glv_gensim,
        word1=word1,
        word2=word2,
        similarity_file=PATH_WORDSIM,
        word2index=word2index,
    )
    return [
        [
            "Skipgram",
            word1,
            word2,
            model_w2v_skipgram_score,
            human_score,
            w2v_skipgram_spearman_corr,
        ],
        [
            "Skipgram (Negative Sampling)",
            word1,
            word2,
            model_w2v_negative_score,
            human_score,
            w2v_negative_spearman_corr,
        ],
        ["GloVe", word1, word2, model_glv_score, human_score, glove_spearman_corr],
        [
            "GloVe (Gensim)",
            word1,
            word2,
            model_glv_gensim_score,
            human_score,
            gensim_spearman_corr,
        ],
    ]


with gr.Blocks() as demo:
    with gr.Tab("Similarities"):
        words_in_similarity_file = set()
        with open(PATH_WORDSIM, "r", encoding="utf-8") as f:
            for line in f:
                w1, w2, _ = line.strip().split()
                words_in_similarity_file.update([w1, w2])

        # Filter words that are also in the Reuters corpus
        valid_words = [word for word in words_in_similarity_file if word in word2index]
        word1_input = None
        word2_input = None

        with gr.Row():
            with gr.Column():
                word1_input = gr.Dropdown(
                    choices=valid_words,
                    label="Word 1 (only the ones available in both corpus and wordsim)",
                )
            with gr.Column():
                word2_input = gr.Dropdown(
                    choices=[],
                    label="Word 2 (auto-populate based on selected Word 1, only ones available in both corpus and wordsim, could be empty!)",
                )

        def update_word2_choices(word1):
            with open(PATH_WORDSIM, "r", encoding="utf-8") as f:
                word2_choices = [
                    w2
                    for w1_, w2, _ in [line.strip().split() for line in f]
                    if w1_ == word1 and w2 in word2index
                ]
                return gr.update(choices=word2_choices)

        word1_input.change(fn=update_word2_choices, inputs=word1_input, outputs=word2_input)
        calculate_button = gr.Button("Calculate")

        result_table = gr.Dataframe(
            headers=["Method", "Word 1", "Word 2", "Model Score", "Human Score", "Spearman Correlation"],
        )

        calculate_button.click(
            fn=event_calc_similarity,
            inputs=[word1_input, word2_input],
            outputs=result_table,
        )

    with gr.Tab("Accuracy"):
        with gr.Row():
            with gr.Column():
                semantic_choices = []
                syntactic_choices = []

                with open("word-test.v1.txt", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    category = None
                    for line in lines:
                        if ":" in line:
                            category = line.strip().split(":")[1].strip()
                        elif category == "capital-common-countries":
                            words = line.strip().split()
                            if len(words) == 4:
                                pair = (words[0], words[1])
                                if pair not in semantic_choices:
                                    semantic_choices.append(pair)
                        elif category == "past-tense":
                            words = line.strip().split()
                            if len(words) == 4:
                                pair = (words[0], words[1])
                                if pair not in syntactic_choices:
                                    syntactic_choices.append(pair)
                ac_word1_input = gr.Dropdown(
                    choices=semantic_choices, label="Word 1", interactive=True
                )

            with gr.Column():
                ac_word2_input = gr.Dropdown(choices=[], label="Word 2", interactive=False)

                def update_ac_word2_choices(ac_word1):
                    with open("word-test.v1.txt", "r", encoding="utf-8") as f:
                        ac_word2_choices = list(
                            set(
                                words[1]
                                for line in f
                                if (words := line.strip().split()) and words[0] == ac_word1
                            )
                        )
                        return gr.update(choices=ac_word2_choices)

                ac_word1_input.change(
                    fn=update_ac_word2_choices,
                    inputs=ac_word1_input,
                    outputs=ac_word2_input,
                )
            with gr.Column():
                ac_word3_input = gr.Dropdown(choices=[], label="Word 3", interactive=True)

                def update_ac_word3_choices(ac_word1, ac_word2):
                    with open("word-test.v1.txt", "r", encoding="utf-8") as f:
                        ac_word3_choices = list(
                            set(
                                words[2]
                                for line in f
                                if (words := line.strip().split())
                                and words[0] == ac_word1
                                and words[1] == ac_word2
                            )
                        )
                        return gr.update(choices=ac_word3_choices)

                ac_word2_input.change(
                    fn=update_ac_word3_choices,
                    inputs=[ac_word1_input, ac_word2_input],
                    outputs=ac_word3_input,
                )
        with gr.Row():
            with gr.Column():
                gr.Textbox(label="Actual", interactive=False)
            with gr.Column():
                gr.Textbox(label="Prediction", interactive=False)


# Launch the Gradio app
demo.launch()
