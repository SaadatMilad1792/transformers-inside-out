########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import math
import torch
import torch.nn as nn
from .token_embeddings import TokenEmbedding

########################################################################################################################
## -- the entire transformer encoder architecture -- ###################################################################
########################################################################################################################
class MultiHeadedAttention(nn.Module):
  def __init__(self, model_emb, num_heads):
    super(MultiHeadedAttention, self).__init__()
    self.model_emb = model_emb
    self.num_heads = num_heads
    self.heads_emb = model_emb // num_heads
    self.qkv_extractor = nn.Linear(model_emb, 3 * model_emb)
    self.qkv_connector = nn.Linear(model_emb, model_emb)

  def scaled_self_attention(self, q, k, v, mask = None):
    att_mat = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    att_mat = torch.softmax((att_mat + mask.unsqueeze(1)) if mask is not None else att_mat, dim = -1)
    return torch.matmul(att_mat, v), att_mat

  def forward(self, x, mask = None):
    batch_size, max_sequence_length, model_emb = x.shape
    qkv_vector = self.qkv_extractor(x)
    qkv_vector = qkv_vector.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.heads_emb)
    qkv_vector = qkv_vector.permute(0, 2, 1, 3)
    q, k, v = qkv_vector.chunk(3, dim = -1)
    qkv_vector, att_mat = self.scaled_self_attention(q, k, v, mask)
    qkv_vector = qkv_vector.permute(0, 2, 1, 3).reshape(batch_size, max_sequence_length, self.model_emb)
    qkv_vector = self.qkv_connector(qkv_vector)
    return qkv_vector, att_mat
  
class LayerNorm(nn.Module):
  def __init__(self, param_dim, epsilon = 1e-6):
    super(LayerNorm, self).__init__()
    self.param_dim = param_dim
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(param_dim))
    self.beta = nn.Parameter(torch.zeros(param_dim))

  def forward(self, x):
    dim = [i for i in range(-1, -len(self.param_dim) - 1, -1)]
    mean = x.mean(dim = dim, keepdim = True)
    var = ((x - mean) ** 2).mean(dim = dim, keepdim = True)
    std = var.sqrt() + self.epsilon
    x_new = (x - mean) / std
    return self.gamma * x_new + self.beta

class FeedForward(nn.Module):
  def __init__(self, model_emb, hidden, dropout_p):
    super(FeedForward, self).__init__()
    self.lin_lay_1 = nn.Linear(model_emb, hidden)
    self.lin_lay_2 = nn.Linear(hidden, model_emb)
    self.dropout_p = nn.Dropout(p = dropout_p)
    self.ReLU = nn.ReLU()
  
  def forward(self, x):
    x = self.lin_lay_1(x)
    x = self.ReLU(x)
    x = self.dropout_p(x)
    x = self.lin_lay_2(x)
    return x

class EncoderUnit(nn.Module):
  def __init__(self, model_emb, num_heads, hidden, dropout_p):
    super(EncoderUnit, self).__init__()
    self.attention = MultiHeadedAttention(model_emb, num_heads)
    self.dropout_1 = nn.Dropout(p = dropout_p)
    self.layer_norm_1 = LayerNorm([model_emb])
    self.feed_forward = FeedForward(model_emb, hidden, dropout_p)
    self.dropout_2 = nn.Dropout(p = dropout_p)
    self.layer_norm_2 = LayerNorm([model_emb])

  def forward(self, x, mask):
    x_skip = x.clone()
    x, att_mat = self.attention(x, mask = mask)
    x = self.dropout_1(x)
    x = self.layer_norm_1(x + x_skip)
    x_skip = x.clone()
    x = self.feed_forward(x)
    x = self.dropout_2(x)
    x = self.layer_norm_2(x + x_skip)
    return x

class EncoderSequential(nn.Sequential):
  def forward(self, *inp):
    x, mask = inp
    for module in self._modules.values():
      x = module(x, mask = mask)
    return x

class TransformerEncoder(nn.Module):
  def __init__(self, model_emb, num_heads, hidden, dropout_p, num_layers, src_stoi,
               max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = 'cpu'):
    super(TransformerEncoder, self).__init__()
    self.embedding = TokenEmbedding(max_sequence_length, model_emb, src_stoi, dropout_p, 
                                      SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = device)
    self.encoder = EncoderSequential(*[EncoderUnit(model_emb, num_heads, hidden, dropout_p) for _ in range(num_layers)])
    
  def forward(self, x, enc_sos_token = True, enc_eos_token = True,  mask = None):
    x = self.embedding(x, sos_token = enc_sos_token, eos_token = enc_eos_token)
    return self.encoder(x, mask)