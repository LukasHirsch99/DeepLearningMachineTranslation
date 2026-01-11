import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for machine translation.
    
    Supports:
    - Self-attention: Q=K=V (encoder/decoder self-attention)
    - Cross-attention: Q from decoder, K=V from encoder
    - Causal masking: For autoregressive decoding
    
    Args:
        d_model (int): Model embedding dimension
        nheads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, d_model: int, nheads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nheads == 0, "d_model must be divisible by nheads"
        
        self.d_model = d_model
        self.nheads = nheads
        self.d_head = d_model // nheads
        self.dropout_p = dropout
        
        # Single packed projection for self-attention efficiency
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len_q, d_model]
            key: [batch, seq_len_kv, d_model]
            value: [batch, seq_len_kv, d_model]
            mask: [seq_len_q, seq_len_kv] or [batch, seq_len_q, seq_len_kv]
            is_causal: Apply causal mask for autoregressive decoding
            
        Returns:
            output: [batch, seq_len_q, d_model]
        """
        B, T, DM = query.size()

        if key is query and query is value:
            # Self-attention: Use packed projection (efficient)
            # (B, T, DM) -> (B, T, 3*DM) -> 3 * (B, T, DM)
            q, k, v = self.qkv_proj(query).split(self.d_model, dim=2)
            
            # Reshape for multi-head: (B, T, DM) -> (B, nh, T, hs)
            q = q.view(B, T, self.nheads, self.d_head).transpose(1, 2)
            k = k.view(B, T, self.nheads, self.d_head).transpose(1, 2)
            v = v.view(B, T, self.nheads, self.d_head).transpose(1, 2)
            
        else:
            # Cross-attention: Split the packed weight and apply to different inputs
            q_weight, k_weight, v_weight = self.qkv_proj.weight.split(self.d_model, dim=0)
            
            q = F.linear(query, q_weight)  # (B, T, DM)
            k = F.linear(key, k_weight)    # (B, T, DM)
            v = F.linear(value, v_weight)  # (B, T, DM)
            seq_len_kv = key.shape[1]
            
            # Reshape for multi-head: query uses T, key/value use seq_len_kv
            q = q.view(B, T, self.nheads, self.d_head).transpose(1, 2)
            k = k.view(B, seq_len_kv, self.nheads, self.d_head).transpose(1, 2)
            v = v.view(B, seq_len_kv, self.nheads, self.d_head).transpose(1, 2)
        
        # Use PyTorch's optimized scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Concatenate heads: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, DM)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, DM)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output
