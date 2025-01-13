import torch
import torch.nn as nn


device = torch.device(
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
)

class Skipgram(nn.Module):  # nn.Module is the base class for all neural network modules in PyTorch

    def __init__(self, vocab_size, embed_size, mode="softmax"):
        super(Skipgram, self).__init__()
        self.mode = mode
        self.embedding_v = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)

    def forward(self, center_words, target_words, all_vocabs, negative_words=None):
        # Create embedding vectors for center words, target words, and all words
        center_embeds = self.embedding_v(center_words)
        target_embeds = self.embedding_u(target_words)
        all_embeds = self.embedding_u(all_vocabs)
        negative_log_likelihood = 0

        if self.mode == "softmax":
            # Dot product between the embeddings of the center word and the context word is computed.
            # This measures how similar the center word is to the context word.
            scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)

            # Dot product between the embeddings of the center word and all words in the vocabulary is computed.
            # This is used to normalize the scores across the entire vocabulary (denominator in the softmax function).
            norm_scores = all_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)

            # Negative log of the softmax probability is taken. This is the loss for a single prediction.
            # The overall loss is the average of these values across all predictions in the batch.
            negative_log_likelihood = (-1) * (
                torch.mean(
                    torch.log(torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1))
                )
            )
        elif self.mode == "negative_sampling":
            # Create embedding vectors for negative words
            negative_embeds = self.embedding_u(negative_words)

            # Compute the dot product between center and target word embeddings
            # positive_socre = torch.sum(center_embeds * target_embeds, dim=1)
            positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
            # Compute the dot product between center and negative word embeddings
            # negative_score = torch.bmm(negative_embeds, center_embeds.unsqueeze(2)).squeeze()
            negative_score = negative_embeds.bmm(center_embeds.transpose(1, 2))

            # Compute the positive loss using the sigmoid function
            positive_loss = torch.log(torch.sigmoid(positive_score))
            # Compute the negative loss using the sigmoid function
            negative_loss = torch.sum(torch.log(torch.sigmoid(-negative_score)), dim=1)

            # Compute the negative log likelihood as the mean of the positive and negative losses
            negative_log_likelihood = -torch.mean(positive_loss + negative_loss)

        return negative_log_likelihood

    def get_embedding(self, word_index):
        return (
            self.embedding_v(torch.tensor([word_index], dtype=torch.long).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        # return self.embedding_v(torch.LongTensor([word_index]).to(device)).detach().cpu().numpy()