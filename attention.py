from torch import nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from math import sqrt

def get_embedding(input_ids:torch.Tensor, model_ckpt:str, config) -> torch.Tensor:
    """
    get token embedding, without position embedding
    """
    '''Embedding init'''
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size) 
    """Embedding Process"""
    inputs_embeds = token_emb(input_ids)
    return inputs_embeds, config

def get_token(text:str, model_ckpt:str) -> torch.Tensor:
    '''Tokenize init'''
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return tokens

def dot_product_attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor) -> torch.Tensor:
    '''get attention'''
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
    # print(f"Scores: {scores}")
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights,value)

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.drop_out = nn.Dropout()

    def forward(self, input_ids):
        seq_lenth = input_ids.size(1)
        position_ids = torch.arange(seq_lenth, dtype=torch.long)
        
        # get token embedding & position embedding
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)

        # combine
        emb = token_emb + position_emb
        emb = self.layer_norm(emb)
        emb = self.drop_out(emb)
        return emb

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value):
        attn_outputs = dot_product_attention(
            self.q(query), self.k(key), self.v(value)
        )
        return attn_outputs
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        x = torch.cat([
            h(query, key, value) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x
        
class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state_1 = self.layer_norm_1(x)
        x = x + self.attention(hidden_state_1, hidden_state_1, hidden_state_1)
        hidden_state_2 = self.layer_norm_2(x)
        x = x + self.feed_forward(hidden_state_2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList(TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers))

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    model = "bert-base-uncased"
    text = "Who are you"
    config = AutoConfig.from_pretrained(model)
    token_id = get_token(text, model).input_ids
    # embedding = get_embedding(token_id, model)
    # embedding = Embedding(config)
    encoder = TransformerEncoder(config)
    encode_output = encoder(token_id)
    print(f"encoder output: {encode_output}\n")