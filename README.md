# Giggle
Streamlit URL: https://st125127-nlp-a1-similar-context.streamlit.app

*Remarks: GloVe Gensim is removed from the online Streamlit app due to its large model size.*

![](images/giggle-in-action.gif)

Credits: Reuters dataset is provided by the NLTK library.

Models are trained on a sample of the Reuters corpus, 100-dimensional embeddings, batch size of 128, and 5 epochs.

# Model Comparison

- Modeling results in modeling--results--allsamples_win2_em100_ep50_b256_lr0.pdf
- Accuracy against word-test.v1.txt

| Model               | Window Size | Training Loss       |  Training Time  | Syntactic Accurary | Semantic Accuracy | Spearman Correlation |
| ------------------- | :---------: | :-----------------: | :-------------: | :----------------: | :---------------: | :------------------: |
| Skipgram            |      2      | 15994   `>>` 5779   |  1.4 sec/epoch  |     0.0000         |     0.0000        |       -0.4524        |
| Skipgram (Negative) |      2      | 1099020 `>>` 160336 |  2.0 sec/epoch  |     0.0000         |     0.0000        |       -0.1667        |
| GloVe               |      2      | 1727    `>>` 479    |  0.6 sec/epoch  |     0.0000         |     0.0000        |       -0.0714        |
| GloVe (Gensim)      |      -      | -                   |  -              |     0.9387         |     0.5545        |        0.6019        |

# Similarity Evaluation

- Similarity against wordsim_similarity_goldstandard.txt

| Model | Skipgram | Skipgram (Negative) | GloVe   | GloVe (Gensim) |
| ----- | :------: | :-----------------: | :-----: | :------------: |
|  MSE  | 105.4528 | 95.4902             | 59.1276 | 27.8562        |

# Similar Context

Finding top 10 most similar words to "dollar" across four models.

| Model | Top Words | Similarity Range |
| ----- | --------- | :--------------: |
| Skipgram | `put`, `owned`, `must`, `taking`, `forecast`, `shareholders`, `deficit`, `drop`, `australia`, `against` | 17 - 20 |
| Skipgram (Negative) | `canada`, `electric`, `buy`, `surplus`, `australian`, `services`, `department`, `income`, `sales`, `feb` | 24 - 29 |
| GloVe | `energy`, `far`, `conrac`, `coffee`, `trust`, `holds`, `bill`, `major`, `proxmire`, `better` | 21 - 29 |
| GloVe (Gensim) | `currency`, `greenback`, `euro`, `dollars`, `currencies`, `yen`, `peso`, `weaker`, `price`, `trading` | 0.6 - 0.7 |

### Observations
- **Skipgram**: These words are loosely connected to financial contexts but aren't directly related to `dollar`.
- **Negative Sampling**: These words are more financially and economically oriented.
- **GloVe**: GloVe captures co-occurrence patterns across the corpus but seems to mix relevant financial terms (`bill`, `trust`) with unrelated ones (`coffee`).
- **Gensim**: Highly relevant words—all directly related to currency and finance.