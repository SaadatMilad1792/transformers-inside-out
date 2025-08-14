# Transformer Decoder
In this part of the tutorial, we will explore the transformer decoder architecture in detail. The transformer decoder also consists of three main submodules: masked multi-head self attention, multi-head cross attention, and feed forward layers, each followed by layer normalization with skip connections. A key difference from the encoder is that the decoder uses masked self attention in its first submodule to ensure that each position can only attend to previous positions, preserving the autoregressive nature of sequence generation. The second submodule, multi-head cross attention, allows the decoder to attend to the encoder’s output, integrating contextual information from the source sequence.

As with the encoder, the transformer decoder module can be stacked multiple times, where `N` represents the number of decoder layers cascaded. At this stage, the inputs to the decoder are typically the embedded and positionally encoded target tokens (shifted right during training), shaped as (`batch_size`, `max_sequence_len`, `embedding_size`). The table below outlines the subsections required to fully understand the transformer decoder:

|     | Topic                                                                                                             | Description                                                                                                     |
|-----|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| 1   | [Masked Self Attention](#masked-self-attention)                                                                   | How masked self attention ensures each position attends only to previous positions in the target sequence.      |
| 2   | [Multi-Headed Masked Self Attention](#multi-headed-masked-self-attention)                                         | How multiple masked attention heads operate in parallel to capture diverse contextual patterns.                 |
| 3   | [Multi-Headed Cross Attention](#multi-headed-cross-attention)                                                     | How the decoder attends to encoder outputs, integrating source sequence context.                                |
| 4   | [Layer Normalization with Skip Connections](#layer-normalization-with-skip-connections)                           | The purpose of layer normalization and skip connections to stabilize and improve gradient flow.                 |
| 5   | [Feed Forward Layer](#feed-forward-layer)                                                                         | The position-wise feed forward network applied after attention mechanisms.                                      |
| 6   | [Combining Submodules to Create the Transformer Decoder](#combining-submodules-to-create-the-transformer-decoder) | How the components are combined to form one decoder layer and how layers are stacked for deeper architectures.  |

## Masked Self Attention
Masked self attention is one of the core components in the transformer decoder. It enables the decoder to learn dependencies between tokens in the target sequence, while ensuring that each position can only attend to its current and previous positions — preventing the model from "cheating" by looking ahead at future tokens. This is achieved through a mechanism called **masked self attention**.

In this process, three separate inputs — **Q** (Query), **K** (Key), and **V** (Value) — are fed into the self attention module along with a **causal mask**. The mask assigns large negative values to positions that should be hidden (future tokens), ensuring they have no influence during decoding. The attention output is calculated using the following formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$

Where:  
- $Q$ represents the queries,  
- $K$ represents the keys,  
- $V$ represents the values,  
- $d_k$ is the dimension of the keys (used to scale the dot products), and  
- $M$ is the **causal mask** used to block attention to future positions.

The softmax function ensures that the attention weights sum to one, allowing the model to focus on the most relevant visible parts of the sequence.

Below is a simple implementation of **masked self attention** in PyTorch:

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
mask = torch.triu(torch.ones(max_sequence_length, max_sequence_length) * float('-inf'), diagonal = 1)
out, att_mat = masked_self_attention(q, k, v, mask)
```

## Multi-Headed Masked Self Attention
Multi-headed attention in the transformer **decoder** has two distinct roles: **masked self attention** and **cross attention**. In masked self attention, the mechanism works similarly to encoder self attention but applies a mask to ensure each position can only attend to previous positions in the output sequence, preventing the model from "seeing the future" during training. In cross attention, the queries come from the decoder's previous layer, while the keys and values come from the encoder output, allowing the decoder to align its current generation step with relevant parts of the source sequence.

In both cases, the multi-headed attention mechanism splits the data into multiple smaller parts (called heads) along the embedding dimension. This allows the model to capture a richer and more diverse set of features from different representation subspaces. Each attention head learns its own attention patterns through separate attention matrices, enabling the model to focus on various aspects of the input simultaneously. Below is the complete implementation of the multi-headed attention mechanism, using the self attention function with an optional mask to support the decoder's masked self attention:

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

## Multi-Headed Cross Attention
Multi-headed cross attention is the only component in the transformer decoder that is fundamentally different from the submodules introduced in the encoder section of this tutorial. In cross attention, the **Query** (`Q`) comes from the previous component in the decoder, while the **Key** (`K`) and **Value** (`V`) are provided by the encoder. This setup allows the decoder to align the target sequence with the relevant parts of the source sequence, effectively learning which words or tokens in the translation correspond to specific elements in the original input.  

The implementation closely mirrors that of traditional multi-headed attention, with one key difference: it processes two separate inputs instead of just one. Here’s how it works:


```python
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
```

## Layer Normalization with Skip Connections
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
The final key component of the transformer decoder is the feed forward layer. It consists of a simple fully connected network with a ReLU activation function and dropout used for regularization. In the original transformer paper, the authors increase the embedding size from 512 to 2048 in this layer before projecting it back down to 512. This expansion allows the model to capture richer and more complex features from the input embeddings. Below is an example implementation of this important part of the transformer decoder:

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

## Combining Submodules to Create the Transformer Decoder
Now, let’s bring together all the components we have covered into a single unit known as the transformer decoder. The implementation below demonstrates how these parts are integrated, including masked self-attention, cross attention, and feed-forward layers. It is important to include skip connections at the appropriate points to help maintain gradient flow and ensure stable, effective training.

```python
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
```

Now in order to test it out for yourself, you can take a quick look at [Transformer Decoder](/development/transformer_decoder_test.ipynb) notebook.

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
src_path, src_name = "../data/dataset/mizan.zip", "en.txt"
tgt_path, tgt_name = "../data/dataset/mizan.zip", "fa.txt"
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

########################################################################################################################
## -- testing the transformer encoder -- ###############################################################################
########################################################################################################################
batch_size, max_sequence_length, model_emb, num_heads, hidden, dropout_p, num_layers = 32, 256, 512, 8, 2048, 0.1, 4
decoder = transformer.TransformerDecoder(model_emb, num_heads, hidden, dropout_p, num_layers, data.tgt_stoi,
                                         max_sequence_length, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device = 'cpu')
x = torch.randn(batch_size, max_sequence_length, model_emb)
en_batch, fa_batch = ("Hi",), ("سلام",)
_, mask, cross_mask = transformer.MaskGenerator(max_sequence_length = max_sequence_length).generate_masks(en_batch, fa_batch)
decoder(x, fa_batch, dec_sos_token = True, dec_eos_token = True, mask = mask, cross_mask = cross_mask).shape
```

And you should see a result similar to this:
```text
torch.Size([1, 256, 512])
```

## Document Navigation
Continue the tutorial by navigating to the previous or next sections, or return to the table of contents using the links below. <br>
[Proceed to the next section: Assembling the Transformer](./Assembling%20the%20Transformer.md) <br>
[Return to the previous section: Transformer Encoder](./Transformer%20Encoder.md) <br>
[Back to the table of contents](/) <br>