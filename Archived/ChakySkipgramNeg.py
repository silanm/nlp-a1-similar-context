import torch
import torch.nn as nn


class SkipgramNegSampling(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(
            vocab_size, emb_size
        )  # center embedding
        self.embedding_u = nn.Embedding(vocab_size, emb_size)  # out embedding
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(
            center_words
        )  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words
        )  # [batch_size, 1, emb_size]
        neg_embeds = -self.embedding_u(
            negative_words
        )  # [batch_size, num_neg, emb_size]

        positive_score = target_embeds.bmm(
            center_embeds.transpose(1, 2)
        ).squeeze(2)
        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        negative_score = neg_embeds.bmm(center_embeds.transpose(1, 2))
        # [batch_size, k, emb_size] @ [batch_size, emb_size, 1] = [batch_size, k, 1]

        loss = self.logsigmoid(positive_score) + torch.sum(
            self.logsigmoid(negative_score), 1
        )

        return -torch.mean(loss)

    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)

        return embeds
