import torch
import torch.nn as nn


class Skipgram(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Skipgram, self).__init__()
        # Initialize the embeddings for center and target words
        self.embedding_v = nn.Embedding(vocab_size, emb_size)
        self.embedding_u = nn.Embedding(vocab_size, emb_size)

    def forward(self, center_words, target_words, all_vocabs):
        # Get the embeddings for center and target words
        center_embeds = self.embedding_v(
            center_words
        )  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words
        )  # [batch_size, 1, emb_size]
        all_embeds = self.embedding_u(
            all_vocabs
        )  # [batch_size, voc_size, emb_size]

        print(
            f"{center_embeds.dim()=}",
            f"{target_embeds.dim()=}",
            f"{all_embeds.dim()=}",
        )

        # Calculate scores by performing batch matrix multiplication
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        # Calculate normalized scores
        norm_scores = all_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, voc_size, emb_size] @ [batch_size, emb_size, 1] = [batch_size, voc_size, 1] = [batch_size, voc_size]

        # Calculate negative log likelihood loss
        nll = -torch.mean(
            torch.log(
                torch.exp(scores)
                / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)
            )
        )  # log-softmax
        # scalar (loss must be scalar)

        return nll  # negative log likelihood
