from torch import nn
import torch
from typing import Optional
from utils.word_embedding import WordEmbedding
from utils.positional_encoding import PositionalEncoding
from utils.attention import MultiHeadAttention
from torch.nn import Transformer
from dataclasses import dataclass


class TransformerEncoderLayer(nn.Module):
    """Encoder layer with self-attention and feed-forward network."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        self.norm1 = nn.LayerNorm(d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model, bias=False)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            src_key_padding_mask: [batch, seq_len] - True where padding
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            is_causal=False,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout1(attn_out)

        # Feed-forward with residual
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout2(ff_out)

        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer with masked self-attention, cross-attention, and feed-forward network."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)

        self.norm1 = nn.LayerNorm(d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model, bias=False)
        self.norm3 = nn.LayerNorm(d_model, bias=False)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, tgt_len, d_model]
            encoder_output: [batch, src_len, d_model]
            tgt_key_padding_mask: [batch, tgt_len] - True where padding in target
            src_key_padding_mask: [batch, src_len] - True where padding in source
        """
        # Masked self-attention on target
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            is_causal=True,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.dropout1(attn_out)

        # Cross-attention to encoder
        x_norm = self.norm2(x)
        cross_out = self.cross_attn(
            x_norm,
            encoder_output,
            encoder_output,
            is_causal=False,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout2(cross_out)

        # Feed-forward
        x_norm = self.norm3(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout3(ff_out)

        return x


@dataclass
class TransformerConfig:
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_len: int = 150


class TranslationTransformer(nn.Module):
    """
    Transformer for sequence-to-sequence machine translation.
    Combines encoder and decoder with embeddings and positional encoding.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        config: TransformerConfig,
        sharedVocab: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.d_model = config.d_model
        self.sharedVocab = sharedVocab

        # Embeddings
        if sharedVocab:
            self.src_embedding = WordEmbedding(
                tgt_vocab_size, config.d_model, padding_idx
            )
            self.tgt_embedding = self.src_embedding
        else:
            self.src_embedding = WordEmbedding(
                src_vocab_size, config.d_model, padding_idx
            )
            self.tgt_embedding = WordEmbedding(
                tgt_vocab_size, config.d_model, padding_idx
            )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_len, dropout=config.dropout
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    config.d_model, config.nhead, config.dim_feedforward, config.dropout
                )
                for _ in range(config.num_encoder_layers)
            ]
        )

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    config.d_model, config.nhead, config.dim_feedforward, config.dropout
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

        self.encoder_norm = nn.LayerNorm(config.d_model, bias=False)
        self.decoder_norm = nn.LayerNorm(config.d_model, bias=False)

        # Output projection
        self.fc_out = nn.Linear(config.d_model, tgt_vocab_size, bias=False)
        self.fc_out.weight = self.tgt_embedding.embedding.weight  # Weight tying

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
            src_mask: [batch, src_len] - True where not padding
            tgt_mask: [batch, tgt_len] - True where not padding

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Embeddings and positional encoding
        src_emb = self.src_embedding(src)
        src_emb = self.positional_encoding(src_emb)
        # src_emb = self.dropout(src_emb)

        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        # tgt_emb = self.dropout(tgt_emb)

        # Encoder
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(
                encoder_output, src_key_padding_mask=src_key_padding_mask
            )

        encoder_output = self.encoder_norm(encoder_output)

        # Decoder
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output,
                encoder_output,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        decoder_output = self.decoder_norm(decoder_output)

        # Output projection
        logits = self.fc_out(decoder_output)

        return logits


class TranslationTransformerPytorch(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        config: TransformerConfig,
        sharedVocab: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.d_model = config.d_model

        # Embeddings
        if sharedVocab:
            self.src_embedding = WordEmbedding(
                tgt_vocab_size, config.d_model, padding_idx
            )
            self.tgt_embedding = self.src_embedding
        else:
            self.src_embedding = WordEmbedding(
                src_vocab_size, config.d_model, padding_idx
            )
            self.tgt_embedding = WordEmbedding(
                tgt_vocab_size, config.d_model, padding_idx
            )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_len, dropout=config.dropout
        )

        self.transformer = Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=False,
        )

        # Output projection
        self.fc_out = nn.Linear(config.d_model, tgt_vocab_size)
        self.fc_out.weight = self.tgt_embedding.embedding.weight  # Weight tying

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
            src_key_padding_mask: [batch, src_len] - True where padding
            tgt_key_padding_mask: [batch, tgt_len] - True where padding

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Embeddings and positional encoding
        src_emb = self.src_embedding(src)
        src_emb = self.positional_encoding(src_emb)
        # src_emb = self.dropout(src_emb)

        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        # tgt_emb = self.dropout(tgt_emb)

        # Build explicit causal mask for target sequence
        tgt_len = tgt_emb.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_len, dtype=tgt_emb.dtype, device=tgt_emb.device
        )

        # Encoder-Decoder with masks
        transformer_out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Output projection
        logits = self.fc_out(transformer_out)

        return logits


if __name__ == "__main__":
    configBig = TransformerConfig(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=150,
    )

    # Initialize the model with larger max_len to handle max_length + special tokens
    # model = TranslationTransformerPytorch( # 90,166,576 parameters
    model = TranslationTransformer(  # 90,213,680 parameters
        src_vocab_size=30_000,
        tgt_vocab_size=30_000,
        config=configBig,
        padding_idx=0,
        sharedVocab=True,
    )

    print(f"Model initialized!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
