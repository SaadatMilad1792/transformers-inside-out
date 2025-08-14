########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import math
import torch
import torch.nn as nn
from .token_embeddings import TokenEmbedding

########################################################################################################################
## -- the entire transformer decoder architecture -- ###################################################################
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
  
class MultiHeadedCrossAttention(nn.Module):
  def __init__(self, model_emb, num_heads):
    super(MultiHeadedCrossAttention, self).__init__()
    self.model_emb = model_emb
    self.num_heads = num_heads
    self.heads_emb = model_emb // num_heads
    self.kv_extractor = nn.Linear(model_emb, 2 * model_emb)
    self.q_extractor = nn.Linear(model_emb, model_emb)
    self.qkv_connector = nn.Linear(model_emb, model_emb)

  def scaled_self_attention(self, q, k, v, mask = None):
    att_mat = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    att_mat = torch.softmax((att_mat + mask.unsqueeze(1)) if mask is not None else att_mat, dim = -1)
    return torch.matmul(att_mat, v), att_mat

  def forward(self, x, y, mask = None):
    batch_size, max_sequence_length, model_emb = x.shape
    kv_vector = self.kv_extractor(x)
    q_vector = self.q_extractor(x)
    kv_vector = kv_vector.reshape(batch_size, max_sequence_length, self.num_heads, 2 * self.heads_emb)
    q_vector = q_vector.reshape(batch_size, max_sequence_length, self.num_heads, self.heads_emb)
    kv_vector = kv_vector.permute(0, 2, 1, 3)
    q = q_vector.permute(0, 2, 1, 3)
    k, v = kv_vector.chunk(2, dim = -1)
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

class DecoderUnit(nn.Module):
  def __init__(self, model_emb, num_heads, hidden, dropout_p):
    super(DecoderUnit, self).__init__()
    self.self_attention = MultiHeadedAttention(model_emb, num_heads)
    self.dropout_1 = nn.Dropout(p = dropout_p)
    self.layer_norm_1 = LayerNorm([model_emb])
    self.cross_attention = MultiHeadedCrossAttention(model_emb, num_heads)
    self.dropout_2 = nn.Dropout(p = dropout_p)
    self.layer_norm_2 = LayerNorm([model_emb])
    self.feed_forward = FeedForward(model_emb, hidden, dropout_p)
    self.dropout_3 = nn.Dropout(p = dropout_p)
    self.layer_norm_3 = LayerNorm([model_emb])

  def forward(self, x, y, mask, cross_mask):
    y_skip = y.clone()
    y, att_mat = self.self_attention(y, mask = mask)
    y = self.dropout_1(y)
    y = self.layer_norm_1(y + y_skip)
    y_skip = y.clone()
    y, cross_att_mat = self.cross_attention(y, x, mask = cross_mask)
    y = self.dropout_2(y)
    y = self.layer_norm_2(y + y_skip)
    y_skip = y.clone()
    y = self.feed_forward(y)
    y = self.dropout_3(y)
    y = self.layer_norm_3(y + y_skip)
    return y

class DecoderSequential(nn.Sequential):
  def forward(self, *inp):
    x, y, mask, cross_mask = inp
    for module in self._modules.values():
      y = module(x, y, mask, cross_mask)
    return y

class TransformerDecoder(nn.Module):
  def __init__(self, model_emb, num_heads, hidden, dropout_p, num_layers, tgt_stoi,
               max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = 'cpu'):
    super(TransformerDecoder, self).__init__()
    self.embedding = TokenEmbedding(max_sequence_length, model_emb, tgt_stoi, dropout_p, 
                                      SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = device)
    self.decoder = DecoderSequential(*[DecoderUnit(model_emb, num_heads, hidden, dropout_p) for _ in range(num_layers)])

  def forward(self, x, y, dec_sos_token = True, dec_eos_token = True, mask = None, cross_mask = None):
    y = self.embedding(y, sos_token = dec_sos_token, eos_token = dec_eos_token)
    return self.decoder(x, y, mask, cross_mask)
  