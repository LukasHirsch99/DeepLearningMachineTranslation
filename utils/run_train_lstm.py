"""
LSTM Encoder and Decoder with Attention Mechanism
For Neural Machine Translation

Implements separate encoder and decoder classes following the architecture
described in Bahdanau et al. (2015) - Neural Machine Translation by Jointly
Learning to Align and Translate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from word_embedding import WordEmbedding


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for sequence-to-sequence models.
    
    Encodes source sequences into a sequence of hidden states that capture
    contextual information from both directions (left-to-right and right-to-left).
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        """
        Initialize LSTM encoder.
        
        Args:
            vocab_size: Size of source vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability (applied between LSTM layers if num_layers > 1)
            padding_idx: Index of padding token in vocabulary
        """
        super(LSTMEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = WordEmbedding(
            vocab_size=vocab_size,
            d_model=embed_dim,
            padding_idx=padding_idx
        )
        
        # Bidirectional LSTM
        # Output will be 2 * hidden_dim because of bidirectionality
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode source sequences.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Actual lengths of source sequences (before padding) [batch_size]
                        If None, assumes no padding
        
        Returns:
            encoder_outputs: All hidden states [batch_size, src_len, 2 * hidden_dim]
            hidden: Final hidden state tuple (h_n, c_n)
                   h_n: [num_layers * 2, batch_size, hidden_dim]
                   c_n: [num_layers * 2, batch_size, hidden_dim]
        """
        # Embed source tokens: [batch_size, src_len, embed_dim]
        embedded = self.dropout(self.embedding(src))
        
        # Pack padded sequences for efficiency (optional but recommended)
        if src_lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            src_lengths_cpu = src_lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                src_lengths_cpu,
                batch_first=True,
                enforce_sorted=False
            )
            # LSTM forward pass
            packed_outputs, hidden = self.lstm(packed)
            # Unpack: [batch_size, src_len, 2 * hidden_dim]
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outputs,
                batch_first=True
            )
        else:
            # Regular LSTM forward pass
            encoder_outputs, hidden = self.lstm(embedded)
        
        # encoder_outputs: [batch_size, src_len, 2 * hidden_dim]
        # hidden: (h_n, c_n) where each is [num_layers * 2, batch_size, hidden_dim]
        
        return encoder_outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    
    Computes attention weights by comparing decoder hidden state with each
    encoder hidden state using a learned alignment model.
    
    Reference: Bahdanau et al. (2015) - "Neural Machine Translation by 
    Jointly Learning to Align and Translate"
    """
    
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int):
        """
        Initialize attention mechanism.
        
        Args:
            encoder_hidden_dim: Dimension of encoder hidden states (2 * hidden_dim for bidirectional)
            decoder_hidden_dim: Dimension of decoder hidden states
            attention_dim: Dimension of attention layer
        """
        super(BahdanauAttention, self).__init__()
        
        # Linear layers for encoder hidden states
        self.encoder_projection = nn.Linear(encoder_hidden_dim, attention_dim)
        
        # Linear layer for decoder hidden state
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim)
        
        # Final layer to compute attention score
        self.attention_score = nn.Linear(attention_dim, 1)
        
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.
        
        Args:
            decoder_hidden: Current decoder hidden state [batch_size, decoder_hidden_dim]
            encoder_outputs: All encoder hidden states [batch_size, src_len, encoder_hidden_dim]
            mask: Padding mask [batch_size, src_len] (True = padding, should be ignored)
        
        Returns:
            context: Attention-weighted context vector [batch_size, encoder_hidden_dim]
            attention_weights: Attention weights [batch_size, src_len]
        """
        batch_size, src_len, encoder_hidden_dim = encoder_outputs.size()
        
        # Project encoder outputs: [batch_size, src_len, attention_dim]
        encoder_proj = self.encoder_projection(encoder_outputs)
        
        # Project decoder hidden state: [batch_size, attention_dim]
        # Then expand to match encoder: [batch_size, 1, attention_dim]
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)
        
        # Additive attention: tanh(W_e * h_encoder + W_d * h_decoder)
        # [batch_size, src_len, attention_dim]
        energy = torch.tanh(encoder_proj + decoder_proj)
        
        # Compute attention scores: [batch_size, src_len, 1]
        attention_scores = self.attention_score(energy)
        
        # Remove last dimension: [batch_size, src_len]
        attention_scores = attention_scores.squeeze(2)
        
        # Apply mask if provided (set padding positions to large negative value)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e10)
        
        # Compute attention weights using softmax: [batch_size, src_len]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector as weighted sum of encoder outputs
        # attention_weights: [batch_size, src_len, 1]
        # encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
        # context: [batch_size, encoder_hidden_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention Mechanism.
    
    Alternative attention mechanism that uses multiplicative scoring.
    Generally faster than Bahdanau attention.
    
    Reference: Luong et al. (2015) - "Effective Approaches to Attention-based
    Neural Machine Translation"
    """
    
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, method: str = 'general'):
        """
        Initialize Luong attention.
        
        Args:
            encoder_hidden_dim: Dimension of encoder hidden states
            decoder_hidden_dim: Dimension of decoder hidden states
            method: Scoring method - 'dot', 'general', or 'concat'
        """
        super(LuongAttention, self).__init__()
        
        self.method = method
        
        if method == 'general':
            self.attention_weight = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        elif method == 'concat':
            self.attention_weight = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
            self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.
        
        Args:
            decoder_hidden: Current decoder hidden state [batch_size, decoder_hidden_dim]
            encoder_outputs: All encoder hidden states [batch_size, src_len, encoder_hidden_dim]
            mask: Padding mask [batch_size, src_len]
        
        Returns:
            context: Context vector [batch_size, encoder_hidden_dim]
            attention_weights: Attention weights [batch_size, src_len]
        """
        batch_size, src_len, _ = encoder_outputs.size()
        
        if self.method == 'dot':
            # decoder_hidden: [batch_size, decoder_hidden_dim]
            # encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            # Assume decoder_hidden_dim == encoder_hidden_dim
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
            
        elif self.method == 'general':
            # Transform encoder outputs
            transformed = self.attention_weight(encoder_outputs)  # [batch_size, src_len, decoder_hidden_dim]
            attention_scores = torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)
            
        elif self.method == 'concat':
            # Expand decoder hidden to match encoder outputs
            decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
            # Concatenate
            combined = torch.cat([encoder_outputs, decoder_expanded], dim=2)
            attention_scores = self.v(torch.tanh(self.attention_weight(combined))).squeeze(2)
        
        # Apply mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e10)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder with Attention Mechanism.
    
    Decodes target sequences using attention over encoder outputs.
    At each timestep, computes attention to get context vector, then
    combines it with current input to generate next token.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_hidden_dim: int,  # 2 * encoder_hidden_dim for bidirectional
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
        attention_type: str = 'bahdanau',
        attention_dim: int = 256
    ):
        """
        Initialize LSTM decoder.
        
        Args:
            vocab_size: Size of target vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            encoder_hidden_dim: Dimension of encoder hidden states (2 * hidden for bidirectional)
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
            padding_idx: Index of padding token
            attention_type: Type of attention - 'bahdanau' or 'luong'
            attention_dim: Dimension of attention layer (only for Bahdanau)
        """
        super(LSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type
        
        # Embedding layer
        self.embedding = WordEmbedding(
            vocab_size=vocab_size,
            d_model=embed_dim,
            padding_idx=padding_idx
        )
        
        # Attention mechanism
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim, attention_dim)
        elif attention_type == 'luong':
            self.attention = LuongAttention(encoder_hidden_dim, hidden_dim, method='general')
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # LSTM (input is embedding + context vector)
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Decoder is unidirectional
        )
        
        # Output projection layer (hidden + context -> vocab)
        self.output_projection = nn.Linear(hidden_dim + encoder_hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            input_token: Current input token [batch_size]
            hidden: Previous hidden state (h, c) from LSTM
            encoder_outputs: All encoder hidden states [batch_size, src_len, encoder_hidden_dim]
            src_mask: Source padding mask [batch_size, src_len]
        
        Returns:
            output: Output logits [batch_size, vocab_size]
            hidden: New hidden state (h, c)
            attention_weights: Attention weights [batch_size, src_len]
        """
        batch_size = input_token.size(0)
        
        # Embed input token: [batch_size, 1, embed_dim]
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        
        # Get current decoder hidden state (from previous timestep)
        # hidden[0] is h, shape: [num_layers, batch_size, hidden_dim]
        # We use the top layer's hidden state for attention
        decoder_hidden = hidden[0][-1]  # [batch_size, hidden_dim]
        
        # Compute attention and context vector
        context, attention_weights = self.attention(decoder_hidden, encoder_outputs, src_mask)
        
        # Concatenate embedding and context: [batch_size, 1, embed_dim + encoder_hidden_dim]
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        # LSTM forward: [batch_size, 1, hidden_dim]
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        
        # Remove sequence dimension: [batch_size, hidden_dim]
        lstm_output = lstm_output.squeeze(1)
        
        # Concatenate LSTM output with context for final prediction
        # [batch_size, hidden_dim + encoder_hidden_dim]
        combined = torch.cat([lstm_output, context], dim=1)
        
        # Project to vocabulary: [batch_size, vocab_size]
        output = self.output_projection(combined)
        
        return output, hidden, attention_weights
    
    def forward(
        self,
        tgt: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_hidden: Tuple[torch.Tensor, torch.Tensor],
        src_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through decoder (for training).
        
        Args:
            tgt: Target sequences [batch_size, tgt_len]
            encoder_outputs: Encoder hidden states [batch_size, src_len, encoder_hidden_dim]
            encoder_hidden: Final encoder hidden state (h, c)
            src_mask: Source padding mask [batch_size, src_len]
            teacher_forcing_ratio: Probability of using ground truth as input (vs. prediction)
        
        Returns:
            outputs: Predicted logits [batch_size, tgt_len, vocab_size]
            attention_weights: All attention weights [batch_size, tgt_len, src_len]
        """
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)
        
        # Initialize hidden state from encoder
        # Encoder hidden: [num_layers * 2, batch_size, hidden_dim] (bidirectional)
        # Decoder needs: [num_layers, batch_size, hidden_dim] (unidirectional)
        hidden = self._init_decoder_hidden(encoder_hidden)
        
        # Store outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size).to(tgt.device)
        attention_weights = torch.zeros(batch_size, tgt_len, encoder_outputs.size(1)).to(tgt.device)
        
        # First input is <SOS> token (tgt[:, 0])
        input_token = tgt[:, 0]
        
        for t in range(tgt_len):
            # Single decoding step
            output, hidden, attn = self.forward_step(input_token, hidden, encoder_outputs, src_mask)
            
            # Store output and attention
            outputs[:, t] = output
            attention_weights[:, t] = attn
            
            # Teacher forcing: use ground truth with probability teacher_forcing_ratio
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if t < tgt_len - 1:  # Don't need next input for last timestep
                if use_teacher_forcing:
                    # Use ground truth
                    input_token = tgt[:, t + 1]
                else:
                    # Use model prediction
                    input_token = output.argmax(dim=1)
        
        return outputs, attention_weights
    
    def _init_decoder_hidden(
        self,
        encoder_hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize decoder hidden state from encoder hidden state.
        
        For bidirectional encoder, we need to combine forward and backward states.
        
        Args:
            encoder_hidden: (h, c) from encoder
                          h, c: [num_layers * 2, batch_size, hidden_dim]
        
        Returns:
            Decoder hidden state (h, c): [num_layers, batch_size, hidden_dim]
        """
        h, c = encoder_hidden
        
        # Reshape from [num_layers * 2, batch_size, hidden_dim]
        # to [num_layers, 2, batch_size, hidden_dim]
        batch_size = h.size(1)
        h = h.view(self.num_layers, 2, batch_size, -1)
        c = c.view(self.num_layers, 2, batch_size, -1)
        
        # Combine forward and backward by averaging (or summing, or using only forward)
        # Here we'll average them
        h = torch.mean(h, dim=1)  # [num_layers, batch_size, hidden_dim]
        c = torch.mean(c, dim=1)
        
        return (h, c)


class TranslationLSTM(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
        attention_type: str = "bahdanau"
    ):
        super().__init__()

        self.encoder = LSTMEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx
        )

        self.decoder = LSTMDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_hidden_dim=2 * hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
            attention_type=attention_type
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ):
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)

        outputs, attention_weights = self.decoder(
            tgt=tgt,
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        return outputs, attention_weights
            
