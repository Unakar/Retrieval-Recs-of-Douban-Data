'''
author: xie tian
llama model with BGE embeddings, designed for personalized recommendation
'''
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Ensure transformers package is up to date for compatibility
bge_model_name = 'BAAI/bge-large-zh'
bge_tokenizer = AutoTokenizer.from_pretrained(bge_model_name)
bge_model = AutoModel.from_pretrained(bge_model_name)

class IDEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.mlp = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()
    
    def forward(self, ids):
        x = self.embedding(ids)
        x = self.mlp(x)
        x = self.activation(x)
        return x

class TagEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Linear(bge_model.config.hidden_size, embedding_dim)
        self.activation = nn.ReLU()
        self.bge_model = bge_model
        self.bge_tokenizer = bge_tokenizer
    
    def forward(self, tags):
        # Tokenize the tags (list of strings)
        inputs = self.bge_tokenizer(tags, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bge_model(**inputs)
            # Use the mean pooling of the last hidden state as the tag embedding
            tag_embeds = outputs.last_hidden_state.mean(dim=1)
        x = self.mlp(tag_embeds)
        x = self.activation(x)
        return x

class CombineEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # MLP to reduce dimensionality after concatenation
        self.mlp = nn.Linear(embedding_dim * 3, embedding_dim)
        self.activation = nn.ReLU()
    
    def forward(self, user_embed, item_embed, tag_embed):
        x = torch.cat((user_embed, item_embed, tag_embed), dim=-1)
        x = self.mlp(x)
        x = self.activation(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / (x.size(-1) ** 0.5))
        x_norm = x / (rms + self.eps)
        return self.scale * x_norm

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch_size, seq_len, num_heads, head_dim]
    q_cos = q * cos - torch.roll(q, shifts=1, dims=-1) * sin
    q_sin = q * sin + torch.roll(q, shifts=1, dims=-1) * cos
    k_cos = k * cos - torch.roll(k, shifts=1, dims=-1) * sin
    k_sin = k * sin + torch.roll(k, shifts=1, dims=-1) * cos
    return q_cos, k_cos

class SimplifiedLlamaDecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.o_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.rms_norm1 = RMSNorm(embedding_dim)
        self.rms_norm2 = RMSNorm(embedding_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim)
        )
    
    def forward(self, x):
        residual = x
        x = self.rms_norm1(x)
        
        # Self-attention
        B, N, C = x.size()
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # RoPE embeddings (simplified)
        cos = sin = 1  # Placeholder for actual RoPE computation
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.o_proj(attn_output)
        
        x = residual + attn_output  # Residual connection
        
        # Feedforward
        residual = x
        x = self.rms_norm2(x)
        x = self.ffn(x)
        x = residual + x  # Residual connection
        
        return x

class llamaXTModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_heads, ffn_dim):
        super().__init__()
        self.user_embedding = IDEmbedding(num_users, embedding_dim)
        self.item_embedding = IDEmbedding(num_items, embedding_dim)
        self.tag_embedding = TagEmbedding(embedding_dim)
        
        self.combine_embeddings = CombineEmbeddings(embedding_dim)
        
        self.rms_norm = RMSNorm(embedding_dim)
        self.transformer_layer = SimplifiedLlamaDecoderLayer(embedding_dim, num_heads, ffn_dim)
        self.final_rms_norm = RMSNorm(embedding_dim)
        
        self.output_layer = nn.Linear(embedding_dim, 1)  # Output rating as a scalar
    
    def forward(self, user_ids, item_ids, tags):
        # Embeddings
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        tag_embed = self.tag_embedding(tags)
        
        # Combine embeddings
        x = self.combine_embeddings(user_embed, item_embed, tag_embed)
        
        # Transformer layer
        x = self.rms_norm(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_layer(x)
        x = x.squeeze(1)    # Remove sequence dimension
        x = self.final_rms_norm(x)
        
        # Output rating
        rating = self.output_layer(x)
        return rating


'''
 use the following code to test the model
 # Define hyperparameters
num_users = 10000      # Total number of users
num_items = 5000       # Total number of items
embedding_dim = 768    # Embedding dimension (match with bge-large-zh if needed)
num_heads = 12         # Number of attention heads
ffn_dim = 3072         # Feedforward network dimension

# Instantiate the model
model = RecommendationModel(num_users, num_items, embedding_dim, num_heads, ffn_dim)

# Example input data
user_ids = torch.tensor([123])            # Example user ID
item_ids = torch.tensor([456])            # Example item ID
tags = ['动作', '剧情']                    # Example item tags (list of strings)

# Forward pass
rating = model(user_ids, item_ids, tags)

print("Predicted Rating:", rating.item())
 '''