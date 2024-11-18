import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import math
from typing import Optional, Tuple



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * self.weight
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim -1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Embedding layers
class IDEmbedding(nn.Module):
    def __init__(self, num_ids, embedding_dim):
        super(IDEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_ids, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)
    
    
# Load bge-large-zh model and tokenizer
tokenizer_bge = AutoTokenizer.from_pretrained("BAAI/bge-large-zh")
model_bge = AutoModel.from_pretrained("BAAI/bge-large-zh")

class TagEmbedding(nn.Module):
    def __init__(self, bge_model, output_dim):
        super(TagEmbedding, self).__init__()
        self.bge_model = bge_model  # Pretrained bge model
        self.mlp = nn.Linear(self.bge_model.config.hidden_size, output_dim)

    def forward(self, tags):
        with torch.no_grad():
            inputs = tokenizer_bge(tags, padding=True, truncation=True, return_tensors="pt").to(next(self.parameters()).device)
            outputs = self.bge_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        embeddings = self.mlp(embeddings)
        return embeddings

class CombineEmbeddings(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CombineEmbeddings, self).__init__()
        self.mlp = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings_list):
        x = torch.cat(embeddings_list, dim=-1)
        x = self.mlp(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, n_heads, rope_theta=10000):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.freqs_cis = None
        self.rope_theta = rope_theta

    def forward(self, x):
        bsz, seqlen, _ = x.size()
        if self.freqs_cis is None or self.freqs_cis.size(0) < seqlen:
            self.freqs_cis = precompute_freqs_cis(self.head_dim, seqlen * 2, self.rope_theta).to(x.device)

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        freqs_cis = self.freqs_cis[:seqlen]
        q, k = apply_rotary_emb(q, k, freqs_cis)

        attn_weights = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.einsum('bhtT,bThd->bthd', attn_weights, v)
        attn_output = attn_output.contiguous().view(bsz, seqlen, -1)
        output = self.wo(attn_output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super(FeedForward, self).__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, rope_theta=10000):
        super(TransformerBlock, self).__init__()
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attention = Attention(dim, n_heads, rope_theta)
        self.feed_forward = FeedForward(dim, 4 * dim)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class RecommendationModel(nn.Module):
    def __init__(
        self, 
        num_users, 
        num_items, 
        embedding_dim, 
        combined_dim, 
        transformer_dim, 
        n_heads, 
        output_dim
    ):
        super(RecommendationModel, self).__init__()
        # Embedding layers for user and item IDs
        self.user_embedding = IDEmbedding(num_users, embedding_dim)
        self.item_embedding = IDEmbedding(num_items, embedding_dim)
        
        # Item tag embedding processor
        self.tag_embedding = TagEmbedding(model_bge, embedding_dim)

        # Combine embeddings
        self.combine_embeddings = CombineEmbeddings(embedding_dim * 3, combined_dim)
        
        # RMSNorm
        self.norm1 = RMSNorm(combined_dim)
        
        # Transformer block
        self.transformer_block = TransformerBlock(combined_dim, n_heads)
        
        # Second RMSNorm
        self.norm2 = RMSNorm(combined_dim)
        
        # Output MLP head
        self.output_head = nn.Linear(combined_dim, output_dim)