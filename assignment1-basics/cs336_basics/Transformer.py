import torch
import torch.nn as nn
from cs336_basics import (
    MultiheadSelfAtten,
    PositionWiseFFN,
    Normalization,
    Embedding,
    Linear,
)
from cs336_basics.SDPA import softmax


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        """
        Initializes a Transformer block with multi-head self-attention and feed-forward network.
        Args:
            d_model (int): The dimension of the input and output features.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the feed-forward network.
            max_seq_len (int): The maximum sequence length.
            theta (int): A parameter for the attention mechanism.
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadSelfAtten.causal_multihead_self_attention(
            d_model,
            num_heads,
            with_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.feed_forward = PositionWiseFFN.PositionWiseFFN(d_model, d_ff)
        self.norm1 = Normalization.RMSNorm(d_model)
        self.norm2 = Normalization.RMSNorm(d_model)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Multi-head self-attention with residual connection and normalization
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output

        # Feed-forward network with residual connection and normalization
        ffn_output = self.feed_forward(self.norm2(x))
        x = x + ffn_output

        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layer: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ):
        """
        Initializes a Transformer-based language model.
        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the input and output features.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the feed-forward network.
            context_length (int): (max_seq_len) The maximum sequence length for RoPE layer.
            theta (int): A parameter for the attention mechanism.
        """
        super(TransformerLM, self).__init__()
        self.embedding = Embedding.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_layer)
        )
        self.output_layer = Linear.Linear(d_model, vocab_size)
        self.norm = Normalization.RMSNorm(d_model)
        self.context_length = context_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer language model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size) containing logits for each token.
        """
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)  # (batch_size, seq_len, d_model)
        x = self.norm(x)  # (batch_size, seq_len, d_model)
        logits = self.output_layer(
            x
        )  # (batch_size, seq_len, vocab_size) # no softmax here
        return logits
