# Transformer Encoder
In this part of the tutorial, we will explore the transformer encoder architecture in detail. The transformer encoder consists of three main submodules: multi head attention, layer normalization with skip connections, and feed forward layers. It is also important to note that the transformer encoder module we implement can be stacked multiple times, where `N` represents the number of encoder layers cascaded. 

At this stage, the inputs have already been processed into a tensor of shape (`batch_size`, `max_sequence_len`, `embedding_size`). To fully understand how the submodules in the encoder work, we need to start by explaining self attention first. The table below shows the subsections needed to fully grasp the transformer encoder:

|     | Topic                                                                                                             | Description                                                                           |
|-----|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 1   | [Self Attention](#self-attention)                                                                                 | Explanation of how self attention captures relationships within the sequence.         |
| 2   | [Multi-Headed Attention](#multi-headed-attention)                                                                 | How multiple attention heads operate in parallel to extract different features.       |
| 3   | [Layer Normalization](#layer-normalization)                                                                       | The purpose of layer normalization and the use of skip connections for stability.     |
| 4   | [Feed Forward Layer](#feed-forward-layer)                                                                         | Description of the position-wise feed forward network after attention.                |
| 5   | [Combining Submodules to Create the Transformer Encoder](#combining-submodules-to-create-the-transformer-encoder) | How all components are combined to form one encoder layer and how layers are stacked. |

## Self Attention
Self attention is one of the core concepts in the transformer architecture. It enables the encoder to learn which parts of a given sequence are related to each other and to what extent. This is achieved through a mechanism called self attention. In this process, three separate inputs — **Q** (Query), **K** (Key), and **V** (Value) — are fed into the self attention module along with an optional mask. The output is calculated using the following formula:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$
Where,  
- $Q$ represents the queries,  
- $K$ represents the keys,  
- $V$ represents the values,  
- $d_k$ is the dimension of the keys (used to scale the dot products), and  
- $M$ is an optional mask added to prevent attention to certain positions.

The softmax function ensures that the attention weights sum to one, allowing the model to focus on the most relevant parts of the sequence. The collowing code block shows the complete implementation of the self attention mechanism with samples to test it with:

```python
import math
import torch

def scaled_self_attention(self, q, k, v, mask = None):
  att_mat = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
  att_mat = torch.softmax((att_mat + mask.unsqueeze(1)) if mask is not None else att_mat, dim = -1)
  return torch.matmul(att_mat, v), att_mat

batch_size, max_sequence_length, model_emb = 1, 4, 8
q = torch.randn(batch_size, max_sequence_length, model_emb)
k = torch.randn(batch_size, max_sequence_length, model_emb)
v = torch.randn(batch_size, max_sequence_length, model_emb)
out, att_mat = scaled_self_attention(q, k, v, mask = None)
```

## Multi-Headed Attention
Multi-headed attention operates similarly to self attention, with the key difference being that it splits the data into multiple smaller parts (called heads) along the embedding dimension. This allows the model to capture a richer and more diverse set of features from different representation subspaces. Each attention head learns its own attention patterns through separate attention matrices, enabling the model to focus on various aspects of the input simultaneously. Below is the complete implementation of the multi-headed attention mechanism, which builds upon the self attention function we developed earlier.

```python
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
```

## Layer Normalization
Layer normalization is a technique used to stabilize and accelerate the training of deep neural networks by normalizing the inputs across the features for each individual data point. Unlike batch normalization, which normalizes across the batch dimension, layer normalization normalizes the activations within each example, making it especially suitable for sequence models like transformers.

The process involves computing the mean and variance of the input features for each data point and then normalizing these features to have zero mean and unit variance. After normalization, two learnable parameters (gamma for scaling and beta for shifting) are applied to allow the model to restore the original distribution if needed. Gamma controls how much the normalized values are scaled, while beta controls the offset added to the scaled values. 

Additionally, skip connections work by adding the original input from an earlier step directly to the output of the current layer. This shortcut path allows gradients to flow more easily during backpropagation, mitigating the vanishing gradient problem. By preserving and blending the initial signal with newly learned transformations, the model can retain important information from previous layers while still benefiting from deeper processing.

By applying layer normalization, the model benefits from more stable gradients and faster convergence during training, which ultimately improves performance and generalization. Here is the code for the layer normalization and how it is applied:

```python
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
```

## Feed Forward Layer
The final key component of the transformer encoder is the feed forward layer. It consists of a simple fully connected network with a ReLU activation function and dropout used for regularization. In the original transformer paper, the authors increase the embedding size from 512 to 2048 in this layer before projecting it back down to 512. This expansion allows the model to capture richer and more complex features from the input embeddings. Below is an example implementation of this important part of the transformer encoder:

```python
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
```

## Combining Submodules to Create the Transformer Encoder
Now, let’s bring together all the components we have covered into a single unit known as the transformer encoder. The implementation below demonstrates how these parts are integrated. It is important to include skip connections at the appropriate points to help prevent vanishing gradient issues and ensure effective training.

```python
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
```

You can find a working example of the multi-layered transformer encoder with Token Embedding layer that takes the source sentence and generates an outout of shape `(batch_size, max_sequence_length, model_emb)`, in [Transformer Encoder](/development/transformer_encoder_test.ipynb) notebook.

```python
########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import os
import sys
sys.path.append(os.path.abspath(".."))
import torch
import transformer

########################################################################################################################
## -- testing the data handler module -- ###############################################################################
########################################################################################################################
src_vocab_path = "../data/vocabs/en_vocab.json"
tgt_vocab_path = "../data/vocabs/fa_vocab.json"
src_path, src_name = "../data/dataset/Tatoeba.zip", "en.txt"
tgt_path, tgt_name = "../data/dataset/Tatoeba.zip", "fa.txt"
SOS_TOKEN, PAD_TOKEN, EOS_TOKEN = '<SOS>', '<PAD>', '<EOS>'

data_handler = transformer.DataHandler(src_path, src_name, src_vocab_path, tgt_path, tgt_name, tgt_vocab_path, 
                                       SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, max_sequence_length = 256, max_sentences = 1000)

data = data_handler.data()

########################################################################################################################
## -- testing the transformer encoder -- ###############################################################################
########################################################################################################################
batch_size, max_sequence_length, model_emb, num_heads, hidden, dropout_p, num_layers = 32, 256, 512, 8, 2048, 0.1, 4

encoder = transformer.TransformerEncoder(model_emb, num_heads, hidden, dropout_p, num_layers, data.src_stoi,
                                         max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = 'cpu')

en_batch, fa_batch = ("Hi",), ("سلام",)
mask, _, _ = transformer.MaskGenerator(max_sequence_length = max_sequence_length).generate_masks(en_batch, fa_batch)
encoder(en_batch, enc_sos_token = True, enc_eof_token = True, mask = mask).shape
```

And here is the output of running the code above:
```text
torch.Size([1, 256, 512])
```

## Document Navigation
Continue the tutorial by navigating to the previous or next sections, or return to the table of contents using the links below. <br>
[Proceed to the next section: Transformer Decoder](./Transformer%20Decoder.md) <br>
[Return to the previous section: Tokenization](./Tokenization.md) <br>
[Back to the table of contents](/) <br>