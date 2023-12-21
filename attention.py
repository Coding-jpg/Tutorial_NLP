from torch import nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from math import sqrt

def get_embedding(text:str, model_ckpt:str) -> torch.Tensor:
    """
    get token and embedding
    """
    '''Tokenize init'''
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    '''Embedding init'''
    config = AutoConfig.from_pretrained(model_ckpt)
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

    """Tokenize & Embedding Process"""

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs_embeds = token_emb(inputs.input_ids)
    # print(f"token: {inputs}\nembedding: {inputs_embeds}\n")


    return inputs_embeds, config

def dot_product_attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor) -> torch.Tensor:
    '''get attention'''
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights,value)

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
        
if __name__ == '__main__':
    model = "bert-base-uncased"
    text = "Who are you"
    embedding, config = get_embedding(text, model) 
    multihead_attn = MultiHeadAttention(config)
    query = key = value = embedding
    attn_output = multihead_attn(query, key, value)
    print(f"attn_output: {attn_output}\nSize:{attn_output.size()}\n")       
