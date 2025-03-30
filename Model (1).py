#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch import Tensor


# In[2]:


# Create causal mask internally
def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # Lower triangular mask (S, S)
    return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, S, S) for broadcasting

# Self-Attention Layer
class SelfAttentionLayer(nn.Module):
    def __init__(self, din, dout, device):
        super().__init__()
        self.wq = nn.Linear(din, dout).to(device)
        self.wk = nn.Linear(din, dout).to(device)
        self.wv = nn.Linear(din, dout).to(device)
        self.scale = torch.sqrt(torch.tensor(dout, dtype=torch.float32)).to(device)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.wq(x)  
        k = self.wk(x)  
        v = self.wv(x)  

        # Compute scaled dot-product attention
        attn_weights = torch.einsum('bqd,bkd->bqk', q, k) / self.scale  # (B, S, S)

        # Apply causal mask
        mask = create_causal_mask(seq_len, x.device)  # Shape: (1, 1, S, S)
        attn_weights = attn_weights.unsqueeze(1)  # Shape: (B, 1, S, S)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf')).squeeze(1)  # (B, S, S)

        # Softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Weighted sum of values
        out = torch.einsum('bqk,bkd->bqd', attn_weights, v)
        return out

# Multi-Head Attention Layer
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads: int, dim: int, device: str):
        super().__init__()
        assert dim % n_heads == 0, "Embedding dim must be divisible by n_heads"
        self.heads = nn.ModuleList([
            SelfAttentionLayer(dim, dim // n_heads, device) \
            for i in range(n_heads)
        ])
        self.wo = nn.Linear(dim, dim).to(device)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=2)
        return self.wo(out)

# Feed Forward Layer
class FeedForwardLayer(nn.Module):
    def __init__(self, dim, device: str):
        super().__init__()
        self.l1 = nn.Linear(dim, dim * 4).to(device)
        self.l2 = nn.Linear(dim * 4, dim).to(device)
        self.gelu = nn.GELU().to(device)

    def forward(self, x):
        return self.l2(self.gelu(self.l1(x)))

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, n_heads, dim, dropout, device):
        super().__init__()
        self.att = MultiHeadAttentionLayer(n_heads, dim, device)
        self.ffl = FeedForwardLayer(dim, device)
        self.norm1 = nn.LayerNorm(dim).to(device)
        self.norm2 = nn.LayerNorm(dim).to(device)
        self.drop1 = nn.Dropout(dropout).to(device)
        self.drop2 = nn.Dropout(dropout).to(device)

    def forward(self, x):
        out = self.norm1(x + self.drop1(self.att(x)))
        return self.norm2(out + self.drop2(self.ffl(out)))

# GPT Model
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, dropout, n_heads, device):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.embeddingpos = nn.Embedding(seq_len, embed_dim).to(device)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, embed_dim, dropout, device) for _ in range(12)
        ])
        self.output = nn.Linear(embed_dim, vocab_size).to(device)

    def forward(self, x):
        batch_size, seq_length = x.shape 
        device = x.device

        we = self.embedding(x)
        positions = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
        pose = self.embeddingpos(positions)
        fembed = we + pose

        for block in self.blocks:
            fembed = block(fembed)

        return self.output(fembed)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




