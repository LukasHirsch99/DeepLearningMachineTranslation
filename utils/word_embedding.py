"""
Word Embedding Layer for Neural Machine Translation

Implements learnable word embeddings that convert token indices into
dense vector representations. These embeddings are learned during training
to capture semantic relationships between words.
"""

import torch
import torch.nn as nn
import math


class WordEmbedding(nn.Module):
    """
    Learnable word embeddings for neural machine translation.

    Converts token indices (integers) into dense vectors (floats) that
    capture semantic meaning. The embeddings are learned during training.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        """
        Initialize embedding layer.

        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            d_model: Dimension of embedding vectors (model dimension)
            padding_idx: Index of padding token (embeddings won't be updated for this)
        """
        super(WordEmbedding, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,  # Ensure padding token does not get updated during training
        )

        # Initialize embeddings with Xavier/Glorot uniform
        # This gives better initial convergence
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to embeddings.

        Args:
            x: Token indices of shape [batch_size, seq_len]

        Returns:
            Embeddings of shape [batch_size, seq_len, d_model]
        """

        # Scale embeddings by sqrt(d_model)
        # This is done in the original Transformer paper
        # Helps balance the magnitude with positional encodings
        return self.embedding(x) * math.sqrt(self.d_model)


class SharedEmbeddings(nn.Module):
    """
    Shared embeddings between encoder and decoder, plus output projection.

    In many NMT systems, we share embeddings across:
    1. Encoder input embeddings
    2. Decoder input embeddings
    3. Decoder output projection (the final linear layer before softmax)

    This reduces parameters and can improve performance when source and
    target languages share vocabulary (or use the same language).
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        """
        Initialize shared embeddings.

        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings
            padding_idx: Padding token index
        """
        super(SharedEmbeddings, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Single embedding table shared across uses
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx
        )

        # Initialize
        nn.init.xavier_uniform_(self.embedding.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)

    def forward(self, x: torch.Tensor, scale: bool = True) -> torch.Tensor:
        """
        Get embeddings for input tokens.

        Args:
            x: Token indices [batch_size, seq_len]
            scale: Whether to scale by sqrt(d_model)

        Returns:
            Embeddings [batch_size, seq_len, d_model]
        """
        embeddings = self.embedding(x)
        if scale:
            embeddings = embeddings * math.sqrt(self.d_model)
        return embeddings

    def get_output_projection(self) -> nn.Linear:
        """
        Create output projection layer that shares weights with embeddings.
        This is called "weight tying" - the embedding matrix is reused
        for the final prediction layer.

        Returns:
            Linear layer with tied weights
        """
        # Create a linear layer
        projection = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Tie its weights to the embedding weights (transpose for correct dimensions)
        projection.weight = self.embedding.weight

        return projection
