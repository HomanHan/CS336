import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding module.
        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of each embedding vector.
            device (torch.device | None): Device on which to store the params.
            dtype (torch.dtype | None): Data type of the params.
        """
        super().__init__()
        self.num_embeds = num_embeddings
        self.embed_dim = embedding_dim
        self.embedding = nn.Parameter(
            torch.empty((self.num_embeds, self.embed_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """

        return self.embedding[token_ids]
