import torch
import torch.nn as nn
from torch.nn import functional as F
import head 
torch.manual_seed(1337)

# Makes a bit of progess -- but does not produce a proper shakesphere model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_head, n_layers):
        super().__init__()
        n_embed = 384
        self.block_size = 256
         # Pytorch will arrange the table as B, T, C)
        # (batch size, block size, channels)
        # This is for a simple Bigram model -- does not take other factors into account
        # self.token_embedding = nn.Embedding(vocab_size, vocab_size)   
        # Has a better embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embed)
    
        # For single self attention layer
        #self.sa_head = head.Head(self.block_size, n_embed)

        # For multiple self attention layer
        #self.sa_head = head.MultiHeadAttention(4, self.block_size, n_embed//4, n_embed)
        #self.ffwd = head.FeedForward(n_embed)
        
        # For blocks of Multiheaded attention
        self.blocks = nn.Sequential(*[head.Block(self.block_size, n_embed, n_head) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)

        # self.blocks = nn.Sequential(
        #     head.Block(self.block_size, n_embed, 4),
        #     head.Block(self.block_size, n_embed, 4),
        #     head.Block(self.block_size, n_embed, 4),
        #     nn.LayerNorm(n_embed)
        # )
        
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, device, targets=None):
        # logits= self.token_embedding(idx)
        _, T = idx.shape

        tok_emb = self.token_embedding(idx)

        # This is the positional embedding -- here it does not matter -- there is a translational invariance
        pos_emb  = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        #s_x = self.sa_head(x)
        #f_x = self.ffwd(s_x)

        s_x = self.blocks(x)
        f_x = self.ln(s_x)
        logits = self.lm_head(f_x)

        if targets == None:
            loss = None

        else:
            # Negative log likelihood loss
            # Trying to call cross entropy in functional form -- which means we do not need to create a module for it
            B, T, C = logits.shape
            #print(logits.shape)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, device):
        for _ in range(max_new_tokens):
            # The valud odf idx cannot be more than the blocksize
            idx_cond = idx[:, -self.block_size:]

            # Get the predictions
            logits, loss = self(idx_cond, device)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Keep remembering everything , but use just the last one.
            idx = torch.cat((idx, idx_next), dim=1)
        return idx 