import torch
import torch.nn as nn


class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(GloVe, self).__init__()
        self.embedding_v = nn.Embedding(
            vocab_size, embed_size
        )  # center embedding
        self.embedding_u = nn.Embedding(vocab_size, embed_size)  # out embedding

        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, coocs, weighting):
        center_embeds = self.embedding_v(
            center_words
        )  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words
        )  # [batch_size, 1, emb_size]

        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(
            center_embeds.transpose(1, 2)
        ).squeeze(2)
        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        # note that coocs already got log
        loss = weighting * torch.pow(
            inner_product + center_bias + target_bias - coocs, 2
        )

        return torch.sum(loss)
