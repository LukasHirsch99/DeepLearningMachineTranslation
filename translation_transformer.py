from torch import nn
import torch
from typing import Optional
from word_embedding import WordEmbedding
from positional_encoding import PositionalEncoding
from attention import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):
    """Encoder layer with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [seq_len, seq_len] or [batch, seq_len, seq_len]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, is_causal=False)
        x = x + self.dropout1(attn_out)
        
        # Feed-forward with residual
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout2(ff_out)
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer with masked self-attention, cross-attention, and feed-forward network."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, tgt_len, d_model]
            encoder_output: [batch, src_len, d_model]
            tgt_mask: Causal mask for target [tgt_len, tgt_len]
            src_mask: Source padding mask [batch, src_len]
        """
        # Masked self-attention on target
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, is_causal=True)
        x = x + self.dropout1(attn_out)
        
        # Cross-attention to encoder
        x_norm = self.norm2(x)
        cross_out = self.cross_attn(x_norm, encoder_output, encoder_output, is_causal=False)
        x = x + self.dropout2(cross_out)
        
        # Feed-forward
        x_norm = self.norm3(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout3(ff_out)
        
        return x


class TranslationTransformer(nn.Module):
    """
    Transformer for sequence-to-sequence machine translation.
    Combines encoder and decoder with embeddings and positional encoding.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = WordEmbedding(src_vocab_size, d_model, padding_idx)
        self.tgt_embedding = WordEmbedding(tgt_vocab_size, d_model, padding_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout=dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
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
            encoder_output = layer(encoder_output)

        encoder_output = self.encoder_norm(encoder_output)
        
        # Decoder
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)
        
        decoder_output = self.decoder_norm(decoder_output)

        # Output projection
        logits = self.fc_out(decoder_output)
        
        return logits