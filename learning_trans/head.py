import torch.nn as nn
import torch
from torch.nn import functional as F

dropout = 0.2

# Puts a section of the network together.
class Block(nn.Module):
    # Put the communication block -- attention and computation block
    # feedforward layer together

    def __init__(self, block_size, n_emb, num_heads):
        super().__init__()
        head_size = n_emb // num_heads
        self.multi_head = MultiHeadAttention(num_heads, block_size, head_size, n_emb)
        self.ffw = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        # Adding the residual path.
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

class FeedForward(nn.Module):
    # Implement a simple feedforward layer
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            # Adding a projection that goes back in the residual pathway
            nn.Linear(4 *n_emb, n_emb),
            nn.Dropout()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x

# Creating Multiple headed attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, block_size, head_size, n_emb):
        super().__init__()
        self.multi_head = nn.ModuleList([Head(block_size, head_size) for _ in range(num_heads)])
        # Projection layer that goes back to the residual pathway
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.multi_head], dim = -1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class Head(nn.Module):
    """ Implement a single head of self attention"""
    def __init__(self,  block_size, n_emb):
        super().__init__()
        self.head_size = n_emb
        self.n_emb = 384
        self.key = nn.Linear(self.n_emb, self.head_size, bias=False) 
        self.query = nn.Linear(self.n_emb, self.head_size, bias=False) 
        self.value = nn.Linear(self.n_emb, self.head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, head_size
        q = self.query(x) # B, T, head_size

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # B, T, T
        # Sice this is the decode block -- we do not consider the future part of the network
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # B, T, head_size
        out = wei @ v # B, T, head_size
        return out
    

