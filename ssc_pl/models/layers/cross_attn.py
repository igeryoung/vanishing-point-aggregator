import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer norm for the residual connection
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        # Linearly project the inputs
        Q = self.q_proj(query)  # (batch_size, query_len, embed_dim)
        K = self.k_proj(key)    # (batch_size, key_len, embed_dim)
        V = self.v_proj(value)  # (batch_size, value_len, embed_dim)

        # Reshape for multi-head attention
        batch_size = Q.size(0)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, query_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, key_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, value_len, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, query_len, key_len)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, query_len, key_len)

        # Apply attention weights to the values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, query_len, head_dim)

        # Reshape back to (batch_size, query_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Apply the output projection
        output = self.out_proj(attn_output)  # (batch_size, query_len, embed_dim)

        # Add residual connection and apply layer normalization
        output = self.layer_norm(output + query)  # (batch_size, query_len, embed_dim)

        return output
