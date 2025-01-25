# Giggle
Streamlit URL: https://st125127-nlp-a1-similar-context.streamlit.app

*Remarks: GloVe Gensim is removed from the online Streamlit app due to its large model size.*

![](images/giggle-in-action.gif)

Models are trained on a sample of the Reuters corpus, 100-dimensional embeddings, batch size of 128, and 5 epochs.

Similarity Scores on Giggle:
- Skipgram and GloVe models may capture co-occurrence patterns but not semantic meaning
- Though the models give high similarity scores but the ouput similar words are still not semantically related to the input word.
- GloVe Gensim provided the most meaningful results in this case e.g., `thin` --> `thick`, `flat`, `thinner`


# Model Comparison

- Modeling results in modeling--results--allsamples_win2_em100_ep50_b256_lr0.pdf
- Accuracy against word-test.v1.txt

| Model               | Window Size | Training Loss       |  Training Time  | Syntactic Accurary | Semantic Accuracy | Spearman Correlation |
| ------------------- | :---------: | :------------------ | :-------------: | :----------------: | :---------------: | :------------------: |
| Skipgram            |      2      | 206,793 `>>` 113,757   |  ~50 sec/epoch  |     -         |     -        |       -0.4524        |
| Skipgram (Negative) |      2      | 35,490,448 `>>` 4,610,617 |  ~95 sec/epoch  |     -         |     -        |       -0.1667        |
| GloVe               |      2      | 25,843 `>>` 68    |  ~24 sec/epoch  |     -         |     -        |       -0.0714        |
| GloVe (Gensim)      |      -      | -                   |  -              |     0.5545         |     0.9387        |        0.6019        |

# Similarity Evaluation

- Similarity against wordsim_similarity_goldstandard.txt

| Model | Skipgram | Skipgram (Negative) | GloVe   | GloVe (Gensim) |
| ----- | :------: | :-----------------: | :-----: | :------------: |
|  MSE  | 24.1058 | 165.4739             | 30.8277 | 27.8562        |

# Similar Context

Finding top 10 most similar words to "dollar" across four models.

| Model | Top Words | Similarity Range |
| ----- | --------- | :--------------: |
| Skipgram | `put`, `owned`, `must`, `taking`, `forecast`, `shareholders`, `deficit`, `drop`, `australia`, `against` | 17 - 20 |
| Skipgram (Negative) | `canada`, `electric`, `buy`, `surplus`, `australian`, `services`, `department`, `income`, `sales`, `feb` | 24 - 29 |
| GloVe | `energy`, `far`, `conrac`, `coffee`, `trust`, `holds`, `bill`, `major`, `proxmire`, `better` | 21 - 29 |
| GloVe (Gensim) | `currency`, `greenback`, `euro`, `dollars`, `currencies`, `yen`, `peso`, `weaker`, `price`, `trading` | 0.6 - 0.7 |

### Observations
- **Skipgram**: These words are loosely connected to financial contexts and aren't directly related to `dollar`.
- **Negative Sampling**: These words are more financially and economically oriented.
- **GloVe**: GloVe captures co-occurrence patterns across the corpus but seems to mix relevant financial terms (`bill`, `trust`) with unrelated ones (`coffee`).
- **Gensim**: Highly relevant words—all directly related to currency and finance.
